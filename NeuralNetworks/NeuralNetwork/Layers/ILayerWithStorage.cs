using NeuralNetwork.Common.Layers;
using NeuralNetwork.Storage;

namespace NeuralNetwork.Layers
{
    internal interface ILayerWithStorage : ILayer
    {
        /// <summary>
        /// Gets the object in which helper matrices are stored.
        /// </summary>
        /// <value>
        /// The helper matrices storage.
        /// </value>
        IMatrixStorage MatrixStorage { get; }
    }
}