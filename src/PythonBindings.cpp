#include "BattleFieldUtils.h"
#include "BattlefieldContext.h"
#include "BattlefieldRenderer.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nanobind/make_iterator.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/bind_vector.h>

SIV3D_SET(s3d::EngineOption::Renderer::Headless)

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
    using FrameArray = nb::ndarray<nb::numpy, std::uint8_t, nb::ndim<3>>;
    using MissileStates = std::vector<toy_acai::MissileState>;
    using HitEvents = std::vector<toy_acai::HitEvent>;

    constexpr const char* Siv3DThreadError = "Siv3D rendering must be used from the thread that created the BattlefieldRenderer";

    [[noreturn]]
    void ThrowSiv3DError(const s3d::Error& error)
    {
        throw std::runtime_error(s3d::Unicode::ToUTF8(error.what()));
    }

    FrameArray MakeFrameArray(const s3d::Image& image)
    {
        const s3d::Size size = image.size();
        const size_t bytes = image.size_bytes();
        auto* values = new std::vector<std::uint8_t>(bytes);
        std::memcpy(values->data(), image.dataAsUint8(), bytes);
        nb::capsule owner(values, [](void* ptr) noexcept
                          { delete static_cast<std::vector<std::uint8_t>*>(ptr); });
        return FrameArray(values->data(), {
            static_cast<size_t>(size.y),
            static_cast<size_t>(size.x),
            size_t{4},
        }, owner);
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

    class FighterStates
    {
    public:
        explicit FighterStates(toy_acai::BattlefieldContext& context)
            : m_context(&context)
        {
        }

        size_t size() const noexcept
        {
            return m_context->fighters.size();
        }

        toy_acai::FighterState& at(nb::ssize_t index)
        {
            if (index < 0)
            {
                index += static_cast<nb::ssize_t>(size());
            }
            if (index < 0 || size() <= static_cast<size_t>(index))
            {
                throw nb::index_error("fighter index out of range");
            }
            return m_context->fighters[static_cast<size_t>(index)];
        }

        void set(nb::ssize_t index, const toy_acai::FighterState& fighter)
        {
            at(index) = fighter;
        }

        auto begin() noexcept
        {
            return m_context->fighters.begin();
        }

        auto end() noexcept
        {
            return m_context->fighters.end();
        }

    private:
        toy_acai::BattlefieldContext* m_context;
    };

    std::array<toy_acai::FighterInput, toy_acai::FighterCount> MakeFighterInputs(nb::iterable inputs)
    {
        std::array<toy_acai::FighterInput, toy_acai::FighterCount> converted{};
        size_t index = 0;
        for (const nb::handle input : inputs)
        {
            if (index == converted.size())
            {
                throw nb::value_error("inputs must contain exactly FIGHTER_COUNT FighterInput values");
            }
            converted[index++] = nb::cast<toy_acai::FighterInput>(input);
        }
        if (index != converted.size())
        {
            throw nb::value_error("inputs must contain exactly FIGHTER_COUNT FighterInput values");
        }
        return converted;
    }

    toy_acai::DistanceFromBoundary ComputeForwardDistanceFromBoundary(
        const toy_acai::BattlefieldContext& context,
        int fighterIndex)
    {
        if (fighterIndex < 0 || toy_acai::FighterCount <= fighterIndex)
        {
            throw nb::index_error("fighter_index out of range");
        }
        return toy_acai::ComputeForwardDistanceFromBoundary(context, fighterIndex);
    }

    struct RendererState
    {
        explicit RendererState()
            : runtime(AcquireSiv3DRuntime())
        {
        }

        std::shared_ptr<Siv3DRuntime> runtime;
        toy_acai::BattlefieldRenderer renderer{};
    };

    class PythonBattlefieldRenderer
    {
    public:
        PythonBattlefieldRenderer()
        {
            try
            {
                m_state = std::make_unique<RendererState>();
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

        ~PythonBattlefieldRenderer() noexcept
        {
            if (m_state && !m_state->runtime->isOwnerThread())
            {
                (void)m_state.release();
            }
        }

        void update(const toy_acai::BattlefieldContext& context, double deltaTime)
        {
            withRenderer([&](toy_acai::BattlefieldRenderer& renderer)
            {
                renderer.update(context, deltaTime);
            });
        }

        void render(const toy_acai::BattlefieldContext& context)
        {
            withRenderer([&](toy_acai::BattlefieldRenderer& renderer)
            {
                renderer.render(context);
            });
        }

        void enableRenderToImageBuffer(s3d::Size size)
        {
            withRenderer([&](toy_acai::BattlefieldRenderer& renderer)
            {
                renderer.EnableRenderToImageBuffer(size);
            });
        }

        FrameArray imageBuffer()
        {
            m_state->runtime->assertOwnerThread();
            try
            {
                return MakeFrameArray(m_state->renderer.imageBuffer());
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

    private:
        template <class Function>
        void withRenderer(Function&& function)
        {
            m_state->runtime->assertOwnerThread();
            try
            {
                std::forward<Function>(function)(m_state->renderer);
            }
            catch (const s3d::Error& error)
            {
                ThrowSiv3DError(error);
            }
        }

        std::unique_ptr<RendererState> m_state;
    };
} // namespace

NB_MODULE(toy_acai_core, m)
{
    m.doc() = "Python bindings for the toy-acai air combat simulator.";

    m.attr("TEAM_COUNT") = toy_acai::TeamCount;
    m.attr("TEAM_FIGHTER_COUNT") = toy_acai::TeamFighterCount;
    m.attr("FIGHTER_COUNT") = toy_acai::FighterCount;
    m.attr("MAX_MISSILE_COUNT") = toy_acai::MaxMissileCount;
    m.attr("FIGHTER_SIZE") = toy_acai::FighterSize;
    m.attr("MISSILE_SIZE") = toy_acai::MissileSize;

    nb::class_<s3d::Vec2>(m, "Vec2")
        .def(nb::init<>())
        .def(nb::init<double, double>(), "x"_a, "y"_a)
        .def_rw("x", &s3d::Vec2::x)
        .def_rw("y", &s3d::Vec2::y);

    nb::class_<s3d::Size>(m, "Size")
        .def(nb::init<>())
        .def(nb::init<s3d::int32, s3d::int32>(), "x"_a, "y"_a)
        .def_rw("x", &s3d::Size::x)
        .def_rw("y", &s3d::Size::y);

    nb::class_<s3d::RectF>(m, "RectF")
        .def(nb::init<>())
        .def(nb::init<double, double, double, double>(), "x"_a, "y"_a, "w"_a, "h"_a)
        .def_rw("x", &s3d::RectF::x)
        .def_rw("y", &s3d::RectF::y)
        .def_rw("w", &s3d::RectF::w)
        .def_rw("h", &s3d::RectF::h)
        .def_prop_rw(
            "pos",
            [](s3d::RectF& rectangle) -> s3d::Vec2& { return rectangle.pos; },
            [](s3d::RectF& rectangle, const s3d::Vec2& position) { rectangle.pos = position; }
        )
        .def_prop_rw(
            "size",
            [](s3d::RectF& rectangle) -> s3d::Vec2& { return rectangle.size; },
            [](s3d::RectF& rectangle, const s3d::Vec2& size) { rectangle.size = size; }
        );

    nb::class_<toy_acai::FighterState>(m, "FighterState")
        .def(nb::init<>())
        .def_rw("team_id", &toy_acai::FighterState::teamId)
        .def_rw("member_id", &toy_acai::FighterState::memberId)
        .def_rw("position", &toy_acai::FighterState::position)
        .def_rw("yaw", &toy_acai::FighterState::yaw)
        .def_rw("speed", &toy_acai::FighterState::speed)
        .def_rw("health", &toy_acai::FighterState::health)
        .def_rw("missile_cooldown", &toy_acai::FighterState::missileCooldown)
        .def_rw("out_of_bounds_time", &toy_acai::FighterState::outOfBoundsTime);

    nb::class_<toy_acai::MissileState>(m, "MissileState")
        .def(nb::init<>())
        .def_rw("id", &toy_acai::MissileState::id)
        .def_rw("position", &toy_acai::MissileState::position)
        .def_rw("yaw", &toy_acai::MissileState::yaw)
        .def_rw("speed", &toy_acai::MissileState::speed)
        .def_rw("age", &toy_acai::MissileState::age)
        .def_rw("lock_lost_time", &toy_acai::MissileState::lockLostTime)
        .def_rw("team_id", &toy_acai::MissileState::teamId)
        .def_rw("shooter_fighter_index", &toy_acai::MissileState::shooterFighterIndex)
        .def_rw("target_fighter_index", &toy_acai::MissileState::targetFighterIndex);

    nb::class_<toy_acai::HitEvent>(m, "HitEvent")
        .def(nb::init<>())
        .def_rw("shooter_fighter_index", &toy_acai::HitEvent::shooterFighterIndex)
        .def_rw("target_fighter_index", &toy_acai::HitEvent::targetFighterIndex);

    nb::class_<toy_acai::FighterInput>(m, "FighterInput")
        .def(nb::init<>())
        .def(nb::init<double, double, bool>(), "acceleration"_a, "turn"_a, "fire"_a)
        .def_rw("acceleration", &toy_acai::FighterInput::acceleration)
        .def_rw("turn", &toy_acai::FighterInput::turn)
        .def_rw("fire", &toy_acai::FighterInput::fire);

    nb::class_<FighterStates>(m, "FighterStates")
        .def("__len__", &FighterStates::size)
        .def("__getitem__", &FighterStates::at, nb::rv_policy::reference_internal)
        .def("__setitem__", &FighterStates::set)
        .def("__iter__", [](FighterStates& fighters)
        {
            return nb::make_iterator<nb::rv_policy::reference>(
                nb::type<FighterStates>(),
                "FighterStatesIterator",
                fighters.begin(),
                fighters.end()
            );
        }, nb::keep_alive<0, 1>());

    nb::bind_vector<MissileStates, nb::rv_policy::reference>(m, "MissileStates");
    nb::bind_vector<HitEvents, nb::rv_policy::reference>(m, "HitEvents");

    nb::class_<toy_acai::BattlefieldContext>(m, "BattlefieldContext")
        .def(nb::init<>())
        .def(nb::init<const toy_acai::BattlefieldContext&>())
        .def_prop_ro("fighters", [](toy_acai::BattlefieldContext& context)
        {
            return FighterStates{context};
        }, nb::keep_alive<0, 1>())
        .def_prop_ro("missiles", [](toy_acai::BattlefieldContext& context) -> MissileStates&
        {
            return context.missiles;
        }, nb::rv_policy::reference_internal)
        .def_prop_ro("hit_events", [](toy_acai::BattlefieldContext& context) -> HitEvents&
        {
            return context.hitEvents;
        }, nb::rv_policy::reference_internal)
        .def_rw("next_missile_id", &toy_acai::BattlefieldContext::nextMissileId)
        .def_rw("screen_size", &toy_acai::BattlefieldContext::screenSize)
        .def_rw("battlefield_area", &toy_acai::BattlefieldContext::battlefieldArea)
        .def_rw("battlefield_diagonal_length", &toy_acai::BattlefieldContext::battlefieldDiagonalLength);

    nb::class_<toy_acai::DistanceFromBoundary>(m, "DistanceFromBoundary")
        .def(nb::init<>())
        .def_rw("distance", &toy_acai::DistanceFromBoundary::distance)
        .def_rw("relative_angle", &toy_acai::DistanceFromBoundary::relativeAngle);

    nb::class_<toy_acai::AbsolutePose>(m, "AbsolutePose")
        .def(nb::init<const toy_acai::FighterState&>())
        .def(nb::init<const toy_acai::MissileState&>())
        .def_rw("position", &toy_acai::AbsolutePose::position)
        .def_rw("yaw", &toy_acai::AbsolutePose::yaw);

    nb::class_<toy_acai::RelativePose>(m, "RelativePose")
        .def(nb::init<>())
        .def_rw("relative_position", &toy_acai::RelativePose::relativePosition)
        .def_rw("relative_yaw", &toy_acai::RelativePose::relativeYaw);

    nb::class_<PythonBattlefieldRenderer>(m, "BattlefieldRenderer")
        .def(nb::init<>())
        .def("update", &PythonBattlefieldRenderer::update, "context"_a, "delta_time"_a)
        .def("render", &PythonBattlefieldRenderer::render, "context"_a)
        .def("enable_render_to_image_buffer", &PythonBattlefieldRenderer::enableRenderToImageBuffer, "size"_a)
        .def("image_buffer", &PythonBattlefieldRenderer::imageBuffer);

    m.def("init_battlefield", &toy_acai::InitBattlefield, "context"_a);
    m.def("update_battlefield", [](toy_acai::BattlefieldContext& context, nb::iterable inputs, double deltaTime)
    {
        toy_acai::UpdateBattlefield(context, MakeFighterInputs(inputs), deltaTime);
    }, "context"_a, "inputs"_a, "delta_time"_a);
    m.def("compute_forward_distance_from_boundary", &ComputeForwardDistanceFromBoundary,
          "context"_a, "fighter_index"_a);
    m.def("compute_relative_pose", &toy_acai::ComputeRelativePose, "from_pose"_a, "to_pose"_a);
}
