#include "Table.h"
#include "Board.h"
#include "ResNetImpl.h"
#include <torch/torch.h>
#include <toml.hpp>
#include <ranges>
#include <numeric>
#include <omp.h>
#include <thread>
#include <atomic>
#include <mutex>
#include <optional>

#include <imgui.h>
#include <imgui-SFML.h>
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>

enum class AppState {
    MENU,
    CONFIG,
    TRAINING,
    PLAYING
};

struct AppConfig {
    std::string database = "qtable.db";
    int boards = 100;
    float epsilon = 0.1f;
    float alpha = 0.1f;
    int batchSize = 32;
    int trainStep = 10;
    std::string savePath = "resnet_model.pt";
    int epochs = 100;

    void load() {
        try {
            const toml::value config = toml::parse("config.toml");
            database = toml::find<std::string>(config, "database");
            boards = toml::find<int>(config, "boards");
            epsilon = toml::find<float>(config, "epsilon");
            alpha = toml::find<float>(config, "alpha");
            batchSize = toml::find<int>(config, "batch_size");
            trainStep = toml::find<int>(config, "train_step");
            savePath = toml::find<std::string>(config, "save_path");
            epochs = toml::find<int>(config, "epochs");
        } catch (...) {}
    }

    void save() {
        toml::value config = toml::table{
            {"database", database},
            {"boards", boards},
            {"epsilon", epsilon},
            {"alpha", alpha},
            {"batch_size", batchSize},
            {"train_step", trainStep},
            {"save_path", savePath},
            {"epochs", epochs}
        };
        std::ofstream out("config.toml");
        out << config;
    }
};

struct TrainingStatus {
    std::atomic<int> currentEpoch{0};
    std::atomic<float> lastLoss{0.0f};
    std::atomic<bool> isRunning{false};
    std::string statusMessage = "Idle";
    std::mutex msgMutex;

    void setMessage(std::string msg) {
        std::lock_guard<std::mutex> lock(msgMutex);
        statusMessage = msg;
    }

    std::string getMessage() {
        std::lock_guard<std::mutex> lock(msgMutex);
        return statusMessage;
    }
};

void trainingWorker(AppConfig config, TrainingStatus& status) {
    status.isRunning = true;
    status.setMessage("Initializing Network...");

    Table table(config.database);
    ResNet network;
    torch::Device device(torch::kCUDA);
    if (!torch::cuda::is_available()) {
        device = torch::Device(torch::kCPU);
    }

    network->to(device);
    torch::nn::MSELoss lossFunction;
    torch::optim::Adam optimizer(network->parameters());

    for (int epoch = 1; epoch <= config.epochs && status.isRunning; epoch++) {
        status.currentEpoch = epoch;
        status.setMessage("Epoch " + std::to_string(epoch) + ": Simulating games...");

        std::vector<Board> boards(config.boards);
        std::atomic<bool> complete = false;
        std::vector<std::vector<torch::Tensor>> states(config.boards);
        std::vector<std::vector<double>> scores(config.boards);

        while (!complete && status.isRunning) {
            complete = true;
            std::vector<std::vector<Move>> moves(config.boards);
            std::vector<int> sizes(config.boards);
            std::atomic<int> num_moves = 0;
            std::vector<int> range(config.boards);
            std::iota(range.begin(), range.end(), 0);

            #pragma omp parallel for
            for (int i = 0; i < config.boards; i++) {
                if (boards[i].getResult() == Result::NONE) {
                    moves[i] = boards[i].getMoves();
                    sizes[i] = moves[i].size();
                    num_moves += moves[i].size();
                    complete = false;
                }
            }

            if (complete) break;

            torch::Tensor evaluationBuffer = torch::zeros({num_moves, 3, 3, 3});
            std::vector<int> offsets(config.boards + 1, 0);
            std::partial_sum(sizes.begin(), sizes.end(), offsets.begin() + 1);

            #pragma omp parallel for
            for (int i = 0; i < config.boards; i++) {
                if (boards[i].getResult() == Result::NONE) {
                    int offset = offsets[i];
                    for (int j = 0; j < sizes[i]; j++) {
                        boards[i].makeMove(moves[i][j]);
                        evaluationBuffer[offset + j] = boards[i].getData();
                        boards[i].undo();
                    }
                }
            }

            torch::NoGradGuard no_grad;
            evaluationBuffer = evaluationBuffer.to(device);
            torch::Tensor evaluation = network->forward(evaluationBuffer);

            #pragma omp parallel for
            for (int i = 0; i < config.boards; i++) {
                if (boards[i].getResult() == Result::NONE) {
                    int offset = offsets[i];
                    int top = torch::rand({1}).item<float>() * sizes[i];
                    if (torch::rand({1}).item<float>() > config.epsilon) {
                        torch::Tensor scores_tensor = evaluation.slice(0, offset, offset + sizes[i]);
                        if (boards[i].getTurn() == Player::cross) {
                            top = torch::argmax(scores_tensor).item<int>();
                        } else {
                            top = torch::argmin(scores_tensor).item<int>();
                        }
                    }
                    boards[i].makeMove(moves[i][top]);
                    states[i].push_back(boards[i].getData().clone());

                    Result outcome = boards[i].getResult();
                    if (outcome != Result::NONE) {
                        double score = 0;
                        if (outcome == Result::CROSS) score = 1;
                        else if (outcome == Result::CIRCLE) score = -1;

                        int numMoves = states[i].size();
                        scores[i] = std::vector<double>(numMoves);
                        for (int k = 0; k < numMoves; k++) {
                            scores[i][k] = score * exp(-(numMoves - 1 - k) * config.alpha);
                        }
                    }
                }
            }
        }

        status.setMessage("Epoch " + std::to_string(epoch) + ": Updating Database...");
        table.updateQ(states, scores);

        if (epoch % config.trainStep == 0) {
            status.setMessage("Epoch " + std::to_string(epoch) + ": Training Network...");
            auto trainingData = table.getDataset(config.batchSize);
            torch::Tensor x = trainingData.first.to(device);
            torch::Tensor y = trainingData.second.view({-1, 1}).to(device);

            optimizer.zero_grad();
            torch::Tensor output = network->forward(x);
            torch::Tensor loss = lossFunction->forward(output, y);
            loss.backward();
            optimizer.step();
            status.lastLoss = loss.item<float>();
        }
    }

    status.setMessage("Saving model...");
    torch::save(network, config.savePath);
    status.isRunning = false;
    status.setMessage("Training Complete.");
}

#include <filesystem>

// ... (existing includes)

int main() {
    sf::RenderWindow window(sf::VideoMode({800, 600}), "Tic-Tac-Toe RL");
    window.setFramerateLimit(60);
    
    // Initialize without loading the default 13px font
    ImGui::SFML::Init(window, false);
    
    // Load a system font at a higher native resolution for crisp rendering
    ImGuiIO& io = ImGui::GetIO();
    ImFont* font = nullptr;
    std::vector<std::string> fontPaths = {
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"
    };

    for (const auto& path : fontPaths) {
        if (std::filesystem::exists(path)) {
            font = io.Fonts->AddFontFromFileTTF(path.c_str(), 36.0f);
            if (font) break;
        }
    }

    if (!font) {
        font = io.Fonts->AddFontDefault();
        io.FontGlobalScale = 2.0f; // Scale up the default font if no TTF found
    } else {
        io.FontGlobalScale = 1.2f; // Slight boost to the high-res TTF
    }
    
    // Rebuild the font texture
    if (!ImGui::SFML::UpdateFontTexture()) {
        std::cerr << "Failed to update font texture" << std::endl;
    }

    AppState state = AppState::MENU;
    AppConfig config;
    config.load();

    TrainingStatus trainStatus;
    std::thread trainThread;

    Board gameBoard;
    ResNet gameNetwork;
    bool networkLoaded = false;
    Player humanPlayer = Player::cross;
    Table table(config.database);

    sf::Clock deltaClock;
    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            ImGui::SFML::ProcessEvent(window, *event);
            if (event->is<sf::Event::Closed>()) window.close();
        }

        ImGui::SFML::Update(window, deltaClock.restart());

        sf::Vector2u windowSize = window.getSize();
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2((float)windowSize.x, (float)windowSize.y));
        ImGui::Begin("Main", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

        if (state == AppState::MENU) {
            float centerX = (float)windowSize.x / 2.0f;
            float startY = (float)windowSize.y / 3.0f;

            ImGui::SetCursorPos(ImVec2(centerX - 125, startY));
            if (ImGui::Button("Start Training", ImVec2(250, 60))) {
                state = AppState::TRAINING;
                if (trainThread.joinable()) trainThread.join();
                trainThread = std::thread(trainingWorker, config, std::ref(trainStatus));
            }
            ImGui::SetCursorPos(ImVec2(centerX - 125, startY + 80));
            if (ImGui::Button("Play Game", ImVec2(250, 60))) {
                try {
                    torch::load(gameNetwork, config.savePath);
                    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
                    gameNetwork->to(device);
                    gameNetwork->eval();
                    networkLoaded = true;
                } catch (...) {
                    networkLoaded = false;
                }
                gameBoard = Board();
                state = AppState::PLAYING;
            }
            ImGui::SetCursorPos(ImVec2(centerX - 125, startY + 160));
            if (ImGui::Button("Configuration", ImVec2(250, 60))) {
                state = AppState::CONFIG;
            }
            ImGui::SetCursorPos(ImVec2(centerX - 125, startY + 240));
            if (ImGui::Button("Exit", ImVec2(250, 60))) {
                window.close();
            }
        } 
        else if (state == AppState::CONFIG) {
            ImGui::Text("Configuration");
            ImGui::Separator();
            char dbPath[256]; strcpy(dbPath, config.database.c_str());
            if (ImGui::InputText("Database Path", dbPath, 256)) config.database = dbPath;
            ImGui::InputInt("Simulation Boards", &config.boards);
            ImGui::SliderFloat("Epsilon (Exploration)", &config.epsilon, 0.0f, 1.0f);
            ImGui::SliderFloat("Alpha (Decay)", &config.alpha, 0.01f, 0.5f);
            ImGui::InputInt("Batch Size", &config.batchSize);
            ImGui::InputInt("Train Every N Epochs", &config.trainStep);
            ImGui::InputInt("Total Epochs", &config.epochs);
            char modelPath[256]; strcpy(modelPath, config.savePath.c_str());
            if (ImGui::InputText("Model Save Path", modelPath, 256)) config.savePath = modelPath;

            if (ImGui::Button("Save & Back")) {
                config.save();
                state = AppState::MENU;
            }
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) {
                config.load();
                state = AppState::MENU;
            }
        }
        else if (state == AppState::TRAINING) {
            ImGui::Text("Training in Progress...");
            ImGui::Separator();
            ImGui::Text("Epoch: %d / %d", (int)trainStatus.currentEpoch, config.epochs);
            ImGui::Text("Last Loss: %.6f", (float)trainStatus.lastLoss);
            ImGui::Text("Status: %s", trainStatus.getMessage().c_str());

            float progress = (float)trainStatus.currentEpoch / config.epochs;
            ImGui::ProgressBar(progress, ImVec2(-1, 0));

            if (!trainStatus.isRunning) {
                if (ImGui::Button("Back to Menu", ImVec2(200, 50))) {
                    if (trainThread.joinable()) trainThread.join();
                    state = AppState::MENU;
                }
            } else {
                if (ImGui::Button("Stop Training", ImVec2(200, 50))) {
                    trainStatus.isRunning = false;
                }
            }
        }
        else if (state == AppState::PLAYING) {
            if (!networkLoaded) {
                ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: Model file not found! Train the model first.");
                if (ImGui::Button("Back to Menu")) state = AppState::MENU;
            } else {
                ImGui::Text("Tic-Tac-Toe Game");
                ImGui::Separator();
                
                if (gameBoard.getResult() == Result::NONE) {
                    ImGui::Text("Turn: %s", gameBoard.getTurn() == Player::cross ? "X" : "O");
                } else {
                    Result res = gameBoard.getResult();
                    if (res == Result::CROSS) ImGui::TextColored(ImVec4(0, 1, 0, 1), "Winner: X!");
                    else if (res == Result::CIRCLE) ImGui::TextColored(ImVec4(1, 0.5, 0, 1), "Winner: O!");
                    else ImGui::TextColored(ImVec4(0.5, 0.5, 0.5, 1), "It's a Draw!");
                }

                float minDim = std::min((float)windowSize.x, (float)windowSize.y);
                float buttonSize = (minDim * 0.6f) / 3.0f;
                float totalWidth = buttonSize * 3.0f + ImGui::GetStyle().ItemSpacing.x * 2.0f;
                float startX = (windowSize.x - totalWidth) / 2.0f;

                torch::Tensor data = gameBoard.getData();
                for (int i = 0; i < 3; i++) {
                    ImGui::SetCursorPosX(startX);
                    for (int j = 0; j < 3; j++) {
                        std::string label = " ";
                        if (data[0][i][j].item<bool>()) label = "X";
                        else if (data[1][i][j].item<bool>()) label = "O";

                        if (ImGui::Button((label + "##" + std::to_string(i*3+j)).c_str(), ImVec2(buttonSize, buttonSize))) {
                            if (gameBoard.getResult() == Result::NONE && label == " ") {
                                Move m; m.row = i; m.col = j; m.player = gameBoard.getTurn();
                                gameBoard.makeMove(m);

                                // AI Turn
                                if (gameBoard.getResult() == Result::NONE) {
                                    auto moves = gameBoard.getMoves();
                                    torch::Tensor buffer = torch::zeros({(long)moves.size(), 3, 3, 3});
                                    for (int k = 0; k < moves.size(); k++) {
                                        gameBoard.makeMove(moves[k]);
                                        buffer[k] = gameBoard.getData();
                                        gameBoard.undo();
                                    }
                                    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
                                    buffer = buffer.to(device);
                                    torch::Tensor output = gameNetwork->forward(buffer);
                                    int best = (gameBoard.getTurn() == Player::cross) ? torch::argmax(output).item<int>() : torch::argmin(output).item<int>();
                                    gameBoard.makeMove(moves[best]);
                                }
                            }
                        }
                        if (j < 2) ImGui::SameLine();
                    }
                }

                if (ImGui::Button("Reset Game")) gameBoard = Board();
                ImGui::SameLine();
                if (ImGui::Button("Back to Menu")) state = AppState::MENU;
            }
        }

        ImGui::End();

        window.clear();
        ImGui::SFML::Render(window);
        window.display();
    }

    if (trainThread.joinable()) {
        trainStatus.isRunning = false;
        trainThread.join();
    }
    ImGui::SFML::Shutdown();
    return 0;
}