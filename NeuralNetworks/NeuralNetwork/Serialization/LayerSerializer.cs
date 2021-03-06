using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork
{
    internal static class LayerSerializer
    {
        public static ISerializedLayer SerializeLayer(ILayer layer)
        {
            var type = layer.GetType().ToString();
            switch (type)
            {
                case "NeuralNetwork.Layers.L2PenaltyLayer":
                    return SerializeL2Layer(layer);
                case "NeuralNetwork.Layers.BasicStandardLayer":
                    return SerializeBasicStandardLayer(layer);
                default:
                    throw new InvalidOperationException("Unknown layer type to serialize");
            }
        }

        public static ISerializedLayer SerializeL2Layer(ILayer layer)
        {
            var L2PenaltyLayer = layer as L2PenaltyLayer;
            return new SerializedL2PenaltyLayer(SerializeBasicStandardLayer(L2PenaltyLayer.UnderlyingLayer), L2PenaltyLayer.Kappa);
        }

        public static ISerializedLayer SerializeBasicStandardLayer(ILayer layer)
        {
            var standardLayer = layer as BasicStandardLayer;
            var bias = new double[standardLayer.Bias.RowCount];
            for (int i = 0; i < standardLayer.Bias.RowCount; ++i)
            {
                bias[i] = standardLayer.Bias[i, 0];
            }
            var weights = standardLayer.Weights.ToArray();

            return new SerializedStandardLayer(bias, weights, standardLayer.Activator.Type, standardLayer.Parameters);
        }
    }
}
