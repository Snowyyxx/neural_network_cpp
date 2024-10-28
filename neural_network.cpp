#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>  // for rand()
#include <ctime>    // for seeding rand()

// sigmoid activation function, used to introduce non-linearity
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// derivative of the sigmoid function, needed for backpropagation
double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

// helper function to generate a random weight between min and max
double randomWeight(double min, double max) {
    return ((double)rand() / RAND_MAX) * (max - min) + min;
}

class neural_network {
private:
    int input_nodes;  // number of input neurons
    int hidden_nodes; // number of hidden neurons
    int output_nodes; // number of output neurons

    // weights and biases
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_hidden;
    std::vector<double> bias_output;

    double learning_rate; // learning rate for training

public:
    // constructor to initialize the neural network with given parameters
    neural_network(int input_nodes, int hidden_nodes, int output_nodes, double learning_rate);

    // forward propagation
    std::vector<double> feedforward(std::vector<double> inputs);

    // backpropagation
    void train(std::vector<double> inputs, std::vector<double> targets);
};

// constructor implementation
neural_network::neural_network(int input_nodes, int hidden_nodes, int output_nodes, double learning_rate) {
    this->input_nodes=input_nodes;
    this->hidden_nodes=hidden_nodes;
    this->output_nodes=output_nodes;
    this->learning_rate=learning_rate;

    // initialize weights and biases with random values
    srand((unsigned int)time(0));  // seed random number generator

    // initialize weights between input and hidden layer
    weights_input_hidden.resize(input_nodes, std::vector<double>(hidden_nodes));
    for(int i=0;i<input_nodes;++i)
        for(int j=0;j<hidden_nodes;++j)
            weights_input_hidden[i][j]=randomWeight(-1.0, 1.0);

    // initialize weights between hidden and output layer
    weights_hidden_output.resize(hidden_nodes, std::vector<double>(output_nodes));
    for(int i=0;i<hidden_nodes;++i)
        for(int j=0;j<output_nodes;++j)
            weights_hidden_output[i][j]=randomWeight(-1.0, 1.0);

    // initialize biases for hidden and output layers
    bias_hidden.resize(hidden_nodes);
    for(int i=0;i<hidden_nodes;++i)
        bias_hidden[i]=randomWeight(-1.0, 1.0);

    bias_output.resize(output_nodes);
    for(int i=0;i<output_nodes;++i)
        bias_output[i]=randomWeight(-1.0, 1.0);
}

// feedforward implementation
std::vector<double> neural_network::feedforward(std::vector<double> inputs) {
    // input to hidden layer
    std::vector<double> hidden(hidden_nodes);
    for(int i=0;i<hidden_nodes;++i) {
        double sum=bias_hidden[i];
        for(int j=0;j<input_nodes;++j) {
            sum+=inputs[j] * weights_input_hidden[j][i];
        }
        hidden[i]=sigmoid(sum);
    }

    // hidden to output layer
    std::vector<double> outputs(output_nodes);
    for(int i=0;i<output_nodes;++i) {
        double sum=bias_output[i];
        for(int j=0;j<hidden_nodes;++j) {
            sum+=hidden[j] * weights_hidden_output[j][i];
        }
        outputs[i]=sigmoid(sum);
    }

    return outputs;
}

// train method for backpropagation
void neural_network::train(std::vector<double> inputs, std::vector<double> targets) {
    // ----------- feedforward ----------- //
    std::vector<double> hidden(hidden_nodes);
    for(int i=0;i<hidden_nodes;++i) {
        double sum=bias_hidden[i];
        for(int j=0;j<input_nodes;++j) {
            sum+=inputs[j] * weights_input_hidden[j][i];
        }
        hidden[i]=sigmoid(sum);
    }

    std::vector<double> outputs(output_nodes);
    for(int i=0;i<output_nodes;++i) {
        double sum=bias_output[i];
        for(int j=0;j<hidden_nodes;++j) {
            sum+=hidden[j] * weights_hidden_output[j][i];
        }
        outputs[i]=sigmoid(sum);
    }

    // ----------- backpropagation ----------- //
    // calculate output errors
    std::vector<double> output_errors(output_nodes);

    for(int i=0;i<output_nodes;++i) {
        output_errors[i]=targets[i] - outputs[i];
    }

    // calculate gradients for weights_hidden_output
    std::vector<double> gradients_output(output_nodes);
    
    for(int i=0;i<output_nodes;++i) {
        double gradient=output_errors[i] * sigmoidDerivative(outputs[i]) * learning_rate;
        gradients_output[i]=gradient;

        // update biases for output layer
        bias_output[i]+=gradient;
    }

    // calculate hidden layer errors
    std::vector<double> hidden_errors(hidden_nodes, 0.0);
    for(int i=0;i<hidden_nodes;++i) {
        double error=0.0;
        for(int j=0;j<output_nodes;++j) {
            error+=output_errors[j] * weights_hidden_output[i][j];
        }
        hidden_errors[i]=error;
    }

    // calculate gradients for weights_input_hidden
    std::vector<double> gradients_hidden(hidden_nodes);
    for(int i=0;i<hidden_nodes;++i) {
        double gradient=hidden_errors[i] * sigmoidDerivative(hidden[i]) * learning_rate;
        gradients_hidden[i]=gradient;

        // update biases for hidden layer
        bias_hidden[i]+=gradient;
    }

    // update weights between hidden and output layers
    for(int i=0;i<hidden_nodes;++i) {
        for(int j=0;j<output_nodes;++j) {
            double delta_weight=gradients_output[j] * hidden[i];
            weights_hidden_output[i][j]+=delta_weight;
        }
    }

    // update weights between input and hidden layers
    for(int i=0;i<input_nodes;++i) {
        for(int j=0;j<hidden_nodes;++j) {
            double delta_weight=gradients_hidden[j] * inputs[i];
            weights_input_hidden[i][j]+=delta_weight;
        }
    }
}

int main() {
    neural_network nn(2, 2, 1, 0.5);

    std::vector<std::vector<double>> training_inputs = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    std::vector<std::vector<double>> training_targets = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    // train the network
    for(int epoch=0;epoch<10000;++epoch) {
        int sample_n=training_inputs.size();
        for(int i=0;i<sample_n;++i) {
            nn.train(training_inputs[i], training_targets[i]);
        }
    }

    // test the network
    std::cout << "testing the neural network on xor problem:\n";
    for(int i=0;i<training_inputs.size();++i) {
        std::vector<double> output = nn.feedforward(training_inputs[i]);
        std::cout << training_inputs[i][0] << " xor " << training_inputs[i][1] << " = " << output[0] << "\n";
    }

    return 0;
}
