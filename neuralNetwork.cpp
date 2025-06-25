#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

enum ActivationFunction {
    SIGMOID,
    TANH
};

// Sigmoid and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}
double sigmoid_derivative(double y) {
    return y * (1.0 - y); // y is sigmoid(x)
}

// Tanh and its derivative
double tanh_activation(double x) {
    return std::tanh(x);
}
double tanh_derivative(double y) {
    return 1.0 - y * y; // y is tanh(x)
}

// Random weight generator
double random_weight(double min = -1.0, double max = 1.0) {
    static std::mt19937 gen(static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count()));
    std::uniform_real_distribution<double> dist(min, max);
    return dist(gen);
}

class NeuralNetwork {
private:
    int input_nodes, hidden_nodes, output_nodes;
    double learning_rate;
    ActivationFunction activation;

    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_hidden;
    std::vector<double> bias_output;

    std::vector<std::vector<double>> prev_weights_input_hidden;
    std::vector<std::vector<double>> prev_weights_hidden_output;

    std::vector<double> hidden_output;

    double activate(double x) const {
        return activation == SIGMOID ? sigmoid(x) : tanh_activation(x);
    }

    double activate_derivative(double y) const {
        return activation == SIGMOID ? sigmoid_derivative(y) : tanh_derivative(y);
    }

public:
    NeuralNetwork(int input, int hidden, int output, double lr, ActivationFunction act = SIGMOID)
        : input_nodes(input), hidden_nodes(hidden), output_nodes(output),
          learning_rate(lr), activation(act) {

        weights_input_hidden.resize(input_nodes, std::vector<double>(hidden_nodes));
        weights_hidden_output.resize(hidden_nodes, std::vector<double>(output_nodes));
        bias_hidden.resize(hidden_nodes);
        bias_output.resize(output_nodes);

        // Random init
        for (auto& row : weights_input_hidden)
            for (auto& val : row) val = random_weight();
        for (auto& row : weights_hidden_output)
            for (auto& val : row) val = random_weight();
        for (auto& val : bias_hidden) val = random_weight();
        for (auto& val : bias_output) val = random_weight();

        // Save initial weights for weight change tracking
        prev_weights_input_hidden = weights_input_hidden;
        prev_weights_hidden_output = weights_hidden_output;
    }

    std::vector<double> feedforward(const std::vector<double>& input) {
        hidden_output.resize(hidden_nodes);
        for (int i = 0; i < hidden_nodes; ++i) {
            double sum = bias_hidden[i];
            for (int j = 0; j < input_nodes; ++j)
                sum += input[j] * weights_input_hidden[j][i];
            hidden_output[i] = activate(sum);
        }

        std::vector<double> output(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            double sum = bias_output[i];
            for (int j = 0; j < hidden_nodes; ++j)
                sum += hidden_output[j] * weights_hidden_output[j][i];
            output[i] = activate(sum);
        }

        return output;
    }

    void train(const std::vector<double>& input, const std::vector<double>& target) {
        auto output = feedforward(input);

        std::vector<double> output_errors(output_nodes);
        std::vector<double> output_gradients(output_nodes);
        for (int i = 0; i < output_nodes; ++i) {
            output_errors[i] = target[i] - output[i];
            output_gradients[i] = output_errors[i] * activate_derivative(output[i]) * learning_rate;
            bias_output[i] += output_gradients[i];
        }

        std::vector<double> hidden_errors(hidden_nodes, 0.0);
        std::vector<double> hidden_gradients(hidden_nodes);
        for (int i = 0; i < hidden_nodes; ++i) {
            for (int j = 0; j < output_nodes; ++j)
                hidden_errors[i] += output_errors[j] * weights_hidden_output[i][j];

            hidden_gradients[i] = hidden_errors[i] * activate_derivative(hidden_output[i]) * learning_rate;
            bias_hidden[i] += hidden_gradients[i];
        }

        for (int i = 0; i < hidden_nodes; ++i)
            for (int j = 0; j < output_nodes; ++j)
                weights_hidden_output[i][j] += output_gradients[j] * hidden_output[i];

        for (int i = 0; i < input_nodes; ++i)
            for (int j = 0; j < hidden_nodes; ++j)
                weights_input_hidden[i][j] += hidden_gradients[j] * input[i];
    }

    void print_weight_change() {
        double total_change = 0.0;

        for (int i = 0; i < input_nodes; ++i)
            for (int j = 0; j < hidden_nodes; ++j)
                total_change += std::abs(weights_input_hidden[i][j] - prev_weights_input_hidden[i][j]);

        for (int i = 0; i < hidden_nodes; ++i)
            for (int j = 0; j < output_nodes; ++j)
                total_change += std::abs(weights_hidden_output[i][j] - prev_weights_hidden_output[i][j]);

        std::cout << "Weight change this epoch: " << total_change << "\n";

        prev_weights_input_hidden = weights_input_hidden;
        prev_weights_hidden_output = weights_hidden_output;
    }
};

int main() {
    NeuralNetwork nn(2, 2, 1, 0.5, SIGMOID);

    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };

    // std::vector<std::vector<double>> targets = {
    //     {0}, {1}, {1}, {0} // XOR
    // };

    // Uncomment for AND or OR
    std::vector<std::vector<double>> targets = { {0}, {0}, {0}, {1} }; // AND
    // std::vector<std::vector<double>> targets = { {0}, {1}, {1}, {1} }; // OR

    const int epochs = 40000;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = nn.feedforward(inputs[i]);
            for (size_t j = 0; j < output.size(); ++j) {
                double error = targets[i][j] - output[j];
                total_loss += error * error;
            }
            nn.train(inputs[i], targets[i]);
        }

        if (epoch % 1000 == 0) {
            std::cout << "Epoch " << epoch << ", MSE Loss: " << total_loss / inputs.size() << "\n";
            nn.print_weight_change();
        }
    }

    std::cout << "\nFinal predictions:\n";
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        auto out = nn.feedforward(inputs[i]);
        int pred = std::round(out[0]);
        std::cout << inputs[i][0] << ", " << inputs[i][1] << " => " << out[0]
                  << " (rounded: " << pred << ", target: " << targets[i][0] << ")\n";
        correct += (pred == static_cast<int>(targets[i][0]));
    }

    std::cout << "Accuracy: " << (100.0 * correct / inputs.size()) << "%\n";
    return 0;
}
