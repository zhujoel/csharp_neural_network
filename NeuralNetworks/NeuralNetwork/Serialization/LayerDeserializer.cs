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
                case LayerType.L2Penalty:
                    var L2Serialized = serializedLayer as SerializedL2PenaltyLayer;
                    return DeserializeL2Layer(L2Serialized, batchSize);
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
                case GradientAdjustmentType.Momentum:
                    var momentum = standardSerialized.GradientAdjustmentParameters as MomentumParameters;
                    return new MomentumLayer(weights, bias, batchSize, activator, momentum);
                default:
                    throw new InvalidOperationException("Unknown Gradient Adjustment Parameter Type");
            }
        }
        private static ILayer DeserializeL2Layer(SerializedL2PenaltyLayer l2Serialized, int batchSize)
        {
            var underlyingSerialized = l2Serialized.UnderlyingSerializedLayer as SerializedStandardLayer;

            var weights = Matrix<double>.Build.DenseOfArray(underlyingSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { underlyingSerialized.Bias });
            var activator = ActivatorFactory.Build(underlyingSerialized.ActivatorType);

            switch (underlyingSerialized.GradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    var learningRate = underlyingSerialized.GradientAdjustmentParameters as FixedLearningRateParameters;
                    return new L2PenaltyLayer(new BasicStandardLayer(weights, bias, batchSize, activator, learningRate), l2Serialized.PenaltyCoefficient) ;
                case GradientAdjustmentType.Momentum:
                    var momentum = underlyingSerialized.GradientAdjustmentParameters as MomentumParameters;
                    return new L2PenaltyLayer(new MomentumLayer(weights, bias, batchSize, activator, momentum), l2Serialized.PenaltyCoefficient);
                default:
                    throw new InvalidOperationException("Unknown Gradient Adjustment Parameter Type");
            }
        }

    }
}