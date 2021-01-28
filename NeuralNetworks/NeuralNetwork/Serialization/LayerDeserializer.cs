using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    internal static class LayerDeserializer
    {
        public static ILayer Deserialize(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    var standardSerialized = serializedLayer as SerializedStandardLayer;
                    return DeserializeStandardLayer(standardSerialized, batchSize);
                case LayerType.Momentum:
                    var momentumSerialized = serializedLayer as SerializedMomentumLayer;
                    return DeserializeMomentumLayer(momentumSerialized, batchSize);

                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");
            }
        }



        private static ILayer DeserializeStandardLayer(SerializedStandardLayer standardSerialized, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfArray(standardSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { standardSerialized.Bias });
            var activator = ActivatorFactory.Build(standardSerialized.ActivatorType);

            switch (standardSerialized.GradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    var learningRate = standardSerialized.GradientAdjustmentParameters as FixedLearningRateParameters;
                    return new BasicStandardLayer(weights, bias, batchSize, activator, learningRate);
                default:
                    throw new InvalidOperationException("Unknown Gradient Adjustment Parameter Type");
            }
        }

        private static ILayer DeserializeMomentumLayer(SerializedMomentumLayer momentumSerialized, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfArray(momentumSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { momentumSerialized.Bias });
            var activator = ActivatorFactory.Build(momentumSerialized.ActivatorType);

            switch (momentumSerialized.GradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.Momentum:
                    var momentumParameter = momentumSerialized.GradientAdjustmentParameters as MomentumParameters;
                    return new MomentumLayer(weights, bias, batchSize, activator, momentumParameter);
                default:
                    throw new InvalidOperationException("Unknown Gradient Adjustment Parameter Type");
            }
        }
    }
}