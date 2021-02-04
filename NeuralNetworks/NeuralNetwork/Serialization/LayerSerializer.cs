using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;

namespace NeuralNetwork
{
    internal static class LayerSerializer
    {
        public static ISerializedLayer SerializeLayer(ILayer layer)
        {
            var L2PenaltyLayer = layer as L2PenaltyLayer;
            var bias = new double[L2PenaltyLayer.UnderlyingLayer.Bias.RowCount];
            for (int i = 0; i < L2PenaltyLayer.UnderlyingLayer.Bias.RowCount; ++i) {
                bias[i] = L2PenaltyLayer.UnderlyingLayer.Bias[i, 0];
            }
            var weights = L2PenaltyLayer.UnderlyingLayer.Weights.ToArray();
            return new SerializedL2PenaltyLayer(new SerializedStandardLayer(bias, weights, L2PenaltyLayer.UnderlyingLayer.Activator.Type, L2PenaltyLayer.UnderlyingLayer.MomentumParameter), L2PenaltyLayer.Kappa);
        }
    }
}
