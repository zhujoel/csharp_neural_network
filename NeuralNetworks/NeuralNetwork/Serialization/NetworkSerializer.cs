using NeuralNetwork.Common.Layers;

namespace NeuralNetwork.Common.Serialization
{
    public static class NetworkSerializer
    {
        public static SerializedNetwork Serialize(INetwork network)
        {
            int layerLength = network.Layers.Length;
            ISerializedLayer[] serializedLayers = new ISerializedLayer[layerLength];
            for (int i = 0; i < layerLength; ++i)
            {
                serializedLayers[i] = LayerSerializer.SerializeLayer(network.Layers[i]);
            }
            return new SerializedNetwork(network.BatchSize, serializedLayers);
        }
    }
}
