using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
namespace NeuralNetwork.Common.Serialization
{
    public class SerializedMomentumLayer : ISerializedLayer
    {
        public SerializedMomentumLayer(double[] bias, double[,] weights, ActivatorType activatorType, IGradientAdjustmentParameters gradientAdjustmentParameters)
        {
            Bias = bias;
            Weights = weights;
            ActivatorType = activatorType;
            GradientAdjustmentParameters = gradientAdjustmentParameters;
        }

        public SerializedMomentumLayer()
        {
        }

        public double[] Bias { get; set; }
        public double[,] Weights { get; set; }
        public ActivatorType ActivatorType { get; set; }
        public IGradientAdjustmentParameters GradientAdjustmentParameters { get; set; }
        public LayerType Type => LayerType.Momentum;
    }
}
