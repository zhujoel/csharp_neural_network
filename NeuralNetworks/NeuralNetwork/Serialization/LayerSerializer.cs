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
            var bias = new double[standardLayer.Bias.RowCount * standardLayer.Bias.ColumnCount];
            int idx = 0;
            for (int i = 0; i < standardLayer.Bias.RowCount; ++i) { 
                for(int j = 0; j < standardLayer.Bias.ColumnCount; ++j)
                {
                    bias[idx++] = standardLayer.Bias[i, j];
                }
            }
            var weights = standardLayer.Weights.ToArray();
            return new SerializedStandardLayer(bias, weights, standardLayer.Activator.Type, standardLayer.LearningParameter);
        }
    }
}
