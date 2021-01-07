namespace NeuralNetwork.Common.Serialization
{
    /// <summary>
    /// Definition of the serialization of a network.
    /// </summary>
    public class SerializedNetwork
    {
        public int BatchSize { get; set; }
        public ISerializedLayer[] SerializedLayers { get; set; }

        public SerializedNetwork(int BatchSize, ISerializedLayer[] SerializedLayers)
        {
            this.BatchSize = BatchSize;
            this.SerializedLayers = SerializedLayers;
        }
    }
}