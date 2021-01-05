using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.GradientAdjustments;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    internal class LayerSerializer : ILayerSerializer
    {
        public GradientAdjustmentParameterExtractor GradientAdjustmentSerializer { get; }

        public LayerSerializer()
        {
            GradientAdjustmentSerializer = new GradientAdjustmentParameterExtractor();
        }

        public ISerializedLayer Serialize(ILayer layer)
        {
            switch (layer)
            {
                case StandardLayer standardLayer:
                    return SerializeStandardLayer(standardLayer);

                case InputStandardizingLayer inputNorm:
                    return SerializeInputStandardizingLayer(inputNorm);

                case DropoutLayer dropout:
                    return SerializeDropoutLayer(dropout);

                case L2PenaltyLayer penalty:
                    return SerializeL2PenaltyLayer(penalty);

                case WeightDecayLayer decay:
                    return SerializeWeightDecayLayer(decay);

                default:
                    throw new InvalidOperationException("Unknown layer type: " + layer.GetType());
            }
        }

        private ISerializedLayer SerializeWeightDecayLayer(WeightDecayLayer decay)
        {
            var underlying = Serialize(decay.UnderlyingLayer);
            return new SerializedWeightDecayLayer(underlying, decay.DecayRate);
        }

        private ISerializedLayer SerializeL2PenaltyLayer(L2PenaltyLayer penalty)
        {
            var underlying = Serialize(penalty.UnderlyingLayer);
            return new SerializedL2PenaltyLayer(underlying, penalty.PenaltyCoefficient);
        }

        private ISerializedLayer SerializeDropoutLayer(DropoutLayer dropout)
        {
            return new SerializedDropoutLayer(dropout.LayerSize, dropout.KeepProbability);
        }

        private ISerializedLayer SerializeInputStandardizingLayer(InputStandardizingLayer inputNorm)
        {
            var mean = inputNorm.Mean;
            var stdDev = inputNorm.StdDev;
            var underlying = Serialize(inputNorm.UnderlyingLayer);
            return new SerializedInputStandardizingLayer(underlying, mean, stdDev);
        }

        private ISerializedLayer SerializeStandardLayer(StandardLayer standardLayer)
        {
            var bias = standardLayer.MatrixStorage.Bias.ToColumnArrays()[0];
            var weights = standardLayer.MatrixStorage.Weights.ToArray();
            var activatorType = standardLayer.Activator.Type;
            var adjustmentParameters = GradientAdjustmentSerializer.ExtractParameters(standardLayer.GradientAdjustment);
            return new SerializedStandardLayer(bias, weights, activatorType, adjustmentParameters);
        }
    }
}