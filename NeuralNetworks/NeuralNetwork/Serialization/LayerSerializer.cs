using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;

namespace NeuralNetwork
{
    internal static class LayerSerializer
    {
        public static ISerializedLayer SerializeLayer(ILayer layer)
        {
            var standardLayer = layer as BasicStandardLayer;
            var bias = new double[standardLayer.Bias.RowCount];
            for (int i = 0; i < standardLayer.Bias.RowCount; ++i) {
                bias[i] = standardLayer.Bias[i, 0];
            }
            var weights = standardLayer.Weights.ToArray();
            return new SerializedStandardLayer(bias, weights, standardLayer.Activator.Type, standardLayer.LearningParameter);
        }
    }
}
