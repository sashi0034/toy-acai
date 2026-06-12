#include "BattlefieldContext.h"
#include "BattlefieldRenderer.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

SIV3D_SET(s3d::EngineOption::Renderer::Headless)

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
    using ActionArray = nb::ndarray<const double, nb::shape<toy_acai::FighterCount, 3>, nb::device::cpu, nb::c_contig>;
    using Matrix = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

    constexpr size_t FighterColumnCount = 9;
    constexpr size_t MissileColumnCount = 8;
    constexpr double SimulationDeltaTime = 1.0 / 60.0;
    constexpr double RenderInterval = 0.1;
    constexpr const char* Siv3DThreadError = "Siv3D rendering must be used from the thread that created the rendering BattlefieldEnv";

    [[noreturn]]
    void ThrowSiv3DError(const s3d::Error& error)
    {
        throw std::runtime_error(s3d::Unicode::ToUTF8(error.what()));
    }

    Matrix MakeMatrix(std::vector<double>* values, size_t rows, size_t cols)
    {
        nb::capsule owner(values, [](void* ptr) noexcept
        {
            delete static_cast<std::vector<double>*>(ptr);
        });
        return Matrix(values->data(), {rows, cols}, owner);
    }

    Matrix MakeFighterMatrix(const toy_acai::BattlefieldContext& context)
    {
        auto* values = new std::vector<double>(toy_acai::FighterCount * FighterColumnCount);
        for (size_t i = 0; i < context.fighters.size(); ++i)
        {
            const auto& fighter = context.fighters[i];
            const size_t offset = i * FighterColumnCount;
            (*values)[offset + 0] = static_cast<double>(fighter.teamId);
            (*values)[offset + 1] = static_cast<double>(fighter.memberId);
            (*values)[offset + 2] = fighter.position.x;
            (*values)[offset + 3] = fighter.position.y;
            (*values)[offset + 4] = fighter.yaw;
            (*values)[offset + 5] = fighter.speed;
            (*values)[offset + 6] = fighter.health;
            (*values)[offset + 7] = fighter.missileCooldown;
            (*values)[offset + 8] = fighter.outOfBoundsTime;
        }
        return MakeMatrix(values, toy_acai::FighterCount, FighterColumnCount);
    }

    Matrix MakeMissileMatrix(const toy_acai::BattlefieldContext& context)
    {
        const size_t rows = context.missiles.size();
        auto* values = new std::vector<double>(std::max<size_t>(1, rows * MissileColumnCount));
        for (size_t i = 0; i < rows; ++i)
        {
            const auto& missile = context.missiles[i];
            const size_t offset = i * MissileColumnCount;
            (*values)[offset + 0] = missile.position.x;
            (*values)[offset + 1] = missile.position.y;
            (*values)[offset + 2] = missile.yaw;
            (*values)[offset + 3] = missile.speed;
            (*values)[offset + 4] = missile.age;
            (*values)[offset + 5] = missile.lockLostTime;
            (*values)[offset + 6] = static_cast<double>(missile.teamId);
            (*values)[offset + 7] = static_cast<double>(missile.targetFighterIndex);
        }
        return MakeMatrix(values, rows, MissileColumnCount);
    }

    class Siv3DRuntime
    {
    public:
        Siv3DRuntime()
            : m_ownerThread(std::this_thread::get_id())
        {
        }

        void assertOwnerThread() const
        {
            if (!isOwnerThread())
            {
                throw std::runtime_error(Siv3DThreadError);
            }
        }

        bool isOwnerThread() const noexcept
        {
            return std::this_thread::get_id() == m_ownerThread;
        }

    private:
        std::thread::id m_ownerThread;
        s3d::MainRuntime m_runtime{};
    };

    std::shared_ptr<Siv3DRuntime> AcquireSiv3DRuntime()
    {
        static std::mutex mutex;
        static std::optional<std::thread::id> ownerThread;
        static std::weak_ptr<Siv3DRuntime> weakRuntime;

        std::lock_guard lock{mutex};

        const std::thread::id currentThread = std::this_thread::get_id();
        if (ownerThread.has_value() && *ownerThread != currentThread)
        {
            throw std::runtime_error(Siv3DThreadError);
        }

        if (auto runtime = weakRuntime.lock())
        {
            runtime->assertOwnerThread();
            return runtime;
        }

        std::shared_ptr<Siv3DRuntime> runtime;
        try
        {
            runtime = std::make_shared<Siv3DRuntime>();
        }
        catch (const s3d::Error& error)
        {
            ThrowSiv3DError(error);
        }

        ownerThread = currentThread;
        weakRuntime = runtime;
        return runtime;
    }

    class RenderSession
    {
    public:
        RenderSession(int renderWidth, int renderHeight, const std::string& gifPath)
            : m_runtime(AcquireSiv3DRuntime()),
              m_size(renderWidth, renderHeight)
        {
            if (renderWidth <= 0 || renderHeight <= 0)
            {
                throw std::invalid_argument("render_width and render_height must be positive");
            }

            try
            {
                resetRenderer();

                if (!gifPath.empty())
                {
                    m_gifWriter.emplace();
                    if (!m_gifWriter->open(s3d::Unicode::FromUTF8(gifPath), m_renderer.imageBuffer().size()))
                    {
                        throw std::runtime_error("failed to open GIF writer: " + gifPath);
                    }
                }
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

        ~RenderSession() noexcept
        {
            if (!isOwnerThread())
            {
                return;
            }

            if (m_gifWriter && m_gifWriter->isOpen())
            {
                (void)m_gifWriter->close();
            }
        }

        void assertOwnerThread() const
        {
            m_runtime->assertOwnerThread();
        }

        bool isOwnerThread() const noexcept
        {
            return m_runtime->isOwnerThread();
        }

        void resetRenderer()
        {
            assertOwnerThread();
            try
            {
                m_renderer = toy_acai::BattlefieldRenderer{};
                m_renderer.EnableRenderToImageBuffer(m_size);
                m_renderTime = 0.0;
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

        void updateAndRenderStep(const toy_acai::BattlefieldContext& context)
        {
            assertOwnerThread();
            try
            {
                m_renderer.update(context, SimulationDeltaTime);
                m_renderTime += SimulationDeltaTime;
                if (m_renderTime + 1e-12 < RenderInterval)
                {
                    return;
                }

                m_renderTime -= RenderInterval;
                m_renderer.render(context);

                if (m_gifWriter && m_gifWriter->isOpen())
                {
                    if (!m_gifWriter->writeFrame(m_renderer.imageBuffer(), s3d::SecondsF{RenderInterval}))
                    {
                        throw std::runtime_error("failed to write GIF frame");
                    }
                }
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

        void closeGif()
        {
            assertOwnerThread();
            try
            {
                if (m_gifWriter && m_gifWriter->isOpen() && !m_gifWriter->close())
                {
                    throw std::runtime_error("failed to close GIF writer");
                }
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

        size_t gifFrameCount() const
        {
            assertOwnerThread();
            return m_gifWriter ? m_gifWriter->frameCount() : 0;
        }

    private:
        std::shared_ptr<Siv3DRuntime> m_runtime;
        s3d::Size m_size;
        toy_acai::BattlefieldRenderer m_renderer{};
        std::optional<s3d::AnimatedGIFWriter> m_gifWriter;
        double m_renderTime{};
    };

    class BattlefieldEnv
    {
    public:
        BattlefieldEnv(bool render = false, int renderWidth = 960, int renderHeight = 540, const std::string& gifPath = "")
        {
            if (!render && !gifPath.empty())
            {
                throw std::invalid_argument("gif_path requires render=True");
            }

            if (render && (renderWidth <= 0 || renderHeight <= 0))
            {
                throw std::invalid_argument("render_width and render_height must be positive");
            }

            toy_acai::InitBattlefield(m_context);

            if (render)
            {
                m_renderSession = std::make_unique<RenderSession>(renderWidth, renderHeight, gifPath);
            }
        }

        ~BattlefieldEnv() noexcept
        {
            if (!m_renderSession)
            {
                return;
            }

            if (m_renderSession->isOwnerThread())
            {
                m_renderSession.reset();
            }
            else
            {
                (void)m_renderSession.release();
            }
        }

        nb::dict reset()
        {
            assertRenderOwnerThread();
            toy_acai::InitBattlefield(m_context);
            if (m_renderSession)
            {
                m_renderSession->resetRenderer();
            }
            return observation();
        }

        nb::dict step(ActionArray actions)
        {
            assertRenderOwnerThread();

            std::array<toy_acai::FighterInput, toy_acai::FighterCount> inputs{};
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                inputs[i] = toy_acai::FighterInput{
                    std::clamp(actions(i, 0), -1.0, 1.0),
                    std::clamp(actions(i, 1), -1.0, 1.0),
                    actions(i, 2) >= 0.5,
                };
            }

            toy_acai::UpdateBattlefield(m_context, inputs, SimulationDeltaTime);
            if (m_renderSession)
            {
                m_renderSession->updateAndRenderStep(m_context);
            }
            return observation();
        }

        void closeGif()
        {
            if (m_renderSession)
            {
                m_renderSession->closeGif();
            }
        }

        size_t gifFrameCount() const
        {
            return m_renderSession ? m_renderSession->gifFrameCount() : 0;
        }

    private:
        void assertRenderOwnerThread() const
        {
            if (m_renderSession)
            {
                m_renderSession->assertOwnerThread();
            }
        }

        nb::dict observation() const
        {
            nb::dict result;
            result["fighters"] = MakeFighterMatrix(m_context);
            result["missiles"] = MakeMissileMatrix(m_context);
            result["screen_size"] = nb::make_tuple(m_context.screenSize.x, m_context.screenSize.y);
            result["battlefield"] = nb::make_tuple(
                m_context.battlefieldArea.x,
                m_context.battlefieldArea.y,
                m_context.battlefieldArea.w,
                m_context.battlefieldArea.h);
            result["fighter_count"] = toy_acai::FighterCount;
            result["fighter_columns"] = FighterColumnCount;
            result["missile_columns"] = MissileColumnCount;
            return result;
        }

        toy_acai::BattlefieldContext m_context{};
        std::unique_ptr<RenderSession> m_renderSession;
    };
}

NB_MODULE(toy_acai_core, m)
{
    m.doc() = "Headless Python bindings for the toy-acai air combat simulator.";
    m.attr("FIGHTER_COUNT") = toy_acai::FighterCount;
    m.attr("TEAM_COUNT") = toy_acai::TeamCount;
    m.attr("TEAM_FIGHTER_COUNT") = toy_acai::TeamFighterCount;
    m.attr("FIGHTER_COLUMNS") = FighterColumnCount;
    m.attr("MISSILE_COLUMNS") = MissileColumnCount;
    m.attr("SIMULATION_DELTA_TIME") = SimulationDeltaTime;
    m.attr("RENDER_INTERVAL") = RenderInterval;

    nb::class_<BattlefieldEnv>(m, "BattlefieldEnv")
        .def(nb::init<bool, int, int, std::string>(), "render"_a = false, "render_width"_a = 960, "render_height"_a = 540, "gif_path"_a = "")
        .def("reset", &BattlefieldEnv::reset)
        .def("step", &BattlefieldEnv::step, "actions"_a)
        .def("close_gif", &BattlefieldEnv::closeGif)
        .def("gif_frame_count", &BattlefieldEnv::gifFrameCount);
}
