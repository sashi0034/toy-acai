# include <Siv3D.hpp> // Siv3D v0.6.16

#include "BattlefieldContext.h"
#include "BattlefieldRenderer.h"

#include <array>

using namespace toy_acai;

#if SIV3D_PLATFORM(LINUX)
SIV3D_SET(EngineOption::Renderer::Headless)
#endif

namespace
{
    void InitializeWindowAndScene()
    {
        Window::SetTitle(U"toy-acai");
        Window::SetStyle(WindowStyle::Sizable);
        Scene::SetResizeMode(ResizeMode::Keep);
        Scene::Resize(1920, 1080);
        Window::Resize(Size{1280, 720});
        System::SetTerminationTriggers(UserAction::CloseButtonClicked);
    }

    void SimulateWithoutVideo()
    {
        float renderTime{};
        constexpr double deltaTime = 1.0 / 60.0;
        constexpr int frameCount = 20;
        constexpr FilePathView outputPath = U"battlefield.gif";

        BattlefieldContext battlefield{};
        InitBattlefield(battlefield);

        BattlefieldRenderer renderer{};
        renderer.EnableRenderToImageBuffer(Size{1920, 1080} / 2);

        AnimatedGIFWriter writer{};
        if (!writer.open(outputPath, renderer.imageBuffer().size()))
        {
            throw Error{U"Failed to open GIF writer: " + String{outputPath}};
        }

        int frame = 0;
        do
        {
            const FighterInput input{
                0.8,
                (frame < frameCount / 2) ? 0.5 : -0.5,
                ((frame + 1) % 5) == 0,
            };
            std::array<FighterInput, FighterCount> inputs{};
            inputs.fill(input);

            UpdateBattlefield(battlefield, inputs, deltaTime);

            renderer.update(battlefield, deltaTime);

            renderTime += deltaTime;
            constexpr double renderInterval = 0.1;
            if (renderTime >= renderInterval)
            {
                renderTime -= renderInterval;
                renderer.render(battlefield);

                if (!writer.writeFrame(renderer.imageBuffer(), SecondsF{deltaTime}))
                {
                    throw Error{U"Failed to write GIF frame: " + Format(frame)};
                }

                frame++;
            }
        } while (frame < frameCount);

        if (!writer.close())
        {
            throw Error{U"Failed to close GIF writer: " + String{outputPath}};
        }
    }
}

void Main()
{
#if SIV3D_PLATFORM(LINUX)
    SimulateWithoutVideo();
    return;
#endif

    InitializeWindowAndScene();

    BattlefieldContext battlefield{};
    InitBattlefield(battlefield);

    BattlefieldRenderer renderer{};

    while (System::Update())
    {
        const FighterInput input{
            static_cast<double>(KeyW.pressed()) - static_cast<double>(KeyS.pressed()),
            static_cast<double>(KeyD.pressed()) - static_cast<double>(KeyA.pressed()),
            KeySpace.pressed(),
        };
        std::array<FighterInput, FighterCount> inputs{};
        inputs.fill(input);

        const double deltaTime = Scene::DeltaTime();

        UpdateBattlefield(battlefield, inputs, deltaTime);

        renderer.update(battlefield, deltaTime);
        renderer.render(battlefield);
    }
}

//
// - Debug ビルド: プログラムの最適化を減らす代わりに、エラーやクラッシュ時に詳細な情報を得られます。
//
// - Release ビルド: 最大限の最適化でビルドします。
//
// - [デバッグ] メニュー → [デバッグの開始] でプログラムを実行すると、[出力] ウィンドウに詳細なログが表示され、エラーの原因を探せます。
//
// - Visual Studio を更新した直後は、プログラムのリビルド（[ビルド]メニュー → [ソリューションのリビルド]）が必要な場合があります。
//
