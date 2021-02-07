using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Gradients
{
    public class FixedLRAdjustment : IGradientAdjustment
    {
        readonly FixedLearningRateParameters LearningRate;

        public IGradientAdjustmentParameters GradientParameter { get => this.LearningRate; }

        public FixedLRAdjustment(FixedLearningRateParameters learningRate)
        {
            this.LearningRate = learningRate;
        }
        public void AdjustWeight(Matrix<double> weight, Matrix<double> gradient)
        {
            weight.Subtract(gradient.Multiply(this.LearningRate.LearningRate), weight);
        }

        public void AdjustBias(Matrix<double> bias, Matrix<double> gradient)
        {
            bias.Subtract(gradient.Multiply(this.LearningRate.LearningRate), bias);
        }
    }
}
