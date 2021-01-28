using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Gradients
{
    internal interface IGradientAdjustment
    {
        Matrix<double> Adjust(Matrix<double> gradient);
    }
}
