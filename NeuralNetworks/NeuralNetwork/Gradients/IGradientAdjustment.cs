using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Gradients
{
    internal interface IGradientAdjustment
    {
        double LearningRate{ get; }

        void AdjustWeight(Matrix<double> weight, Matrix<double> gradient);
        void AdjustBias(Matrix<double> bias, Matrix<double> gradient);

    }
}
