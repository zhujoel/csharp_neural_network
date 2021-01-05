using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Storage;
using System;

namespace NeuralNetwork.Layers
{
    internal class DropoutLayer : ILayerWithMode, ILayerWithStorage, IEquatable<DropoutLayer>
    {
        private int batchSize;
        private Mode mode;
        public double KeepProbability { get; }
        public IMatrixStorage MatrixStorage { get; }
        public IDropoutMask Mask { get; }

        public int LayerSize { get; }

        public int InputSize { get; }

        public Mode Mode
        {
            get => mode;
            set
            {
                mode = value;
            }
        }

        public int BatchSize
        {
            get => batchSize;
            set
            {
                batchSize = value;
                Mask.Resize(value);
                MatrixStorage.BatchSize = value;
            }
        }

        public Matrix<double> Activation => MatrixStorage.Activation;

        public Matrix<double> WeightedError => MatrixStorage.WeightedError;

        internal DropoutLayer(IDropoutMask mask, int batchSize)
        {
            Mask = mask;
            LayerSize = Mask.LayerSize;
            InputSize = mask.LayerSize;
            var dummyWeights = Matrix<double>.Build.Dense(InputSize, LayerSize);
            var dummyBias = Matrix<double>.Build.Dense(InputSize, 1);
            MatrixStorage = new MatrixStorage(dummyWeights, dummyBias, batchSize);
            BatchSize = batchSize;
            KeepProbability = mask.KeepProbability;
        }

        public DropoutLayer(int layerSize, int batchSize, Random rng, double keepProbability) :
            this(new DropoutMask(rng, layerSize, keepProbability), batchSize)
        { }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Mask.Value.PointwiseMultiply(upstreamWeightedErrors, MatrixStorage.WeightedError);
        }

        public void UpdateParameters()
        {
            //ntbd
        }

        public bool Equals(DropoutLayer other)
        {
            var sameTrainingBatchSize = BatchSize == other.BatchSize;
            var sameInputSize = InputSize == other.InputSize;
            var sameLayerSize = LayerSize == other.LayerSize;
            var sameProbability = KeepProbability == other.KeepProbability;
            return sameTrainingBatchSize && sameInputSize && sameLayerSize && sameProbability;
        }

        public bool Equals(ILayer other)
        {
            return other is DropoutLayer tstOther && Equals(tstOther);
        }

        public void Propagate(Matrix<double> input)
        {
            Mask.Update(Mode);
            input.PointwiseMultiply(Mask.Value, MatrixStorage.Activation);
        }
    }
}