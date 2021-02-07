using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Gradients
{
    public class FixedLRAdjustment : IGradientAdjustment
    {
        public double LearningRate;

        double IGradientAdjustment.LearningRate => LearningRate;

        public FixedLRAdjustment(FixedLearningRateParameters learningRate)
        {
            this.LearningRate = learningRate.LearningRate;
        }
        public void AdjustWeight(Matrix<double> weight, Matrix<double> gradient)
        {
            weight.Subtract(gradient.Multiply(this.LearningRate), weight);
        }

        public void AdjustBias(Matrix<double> bias, Matrix<double> gradient)
        {
            bias.Subtract(gradient, bias);
        }
    }
}
