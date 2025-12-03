using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace MLNET_SAACLENDATASET
{
    class Program
    {
        static void Main()
        {
            const string fileInputPath = "puntos.csv";

            var mlContext = new MLContext();

            IDataView data = mlContext.Data.LoadFromTextFile<Point>(path: fileInputPath, separatorChar: ',', hasHeader: true);
            var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: ["X", "Y",])
                    .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: 3));

            var model = pipeline.Fit(splitData.TrainSet);

            var predictions = model.Transform(splitData.TestSet);

            ClusteringMetrics metrics = mlContext.Clustering.Evaluate(
                data: predictions,
                scoreColumnName: "Score",
                featureColumnName:"Features");

            // Cuanto menor mejor, la media de los ejemplos a sus respectivos cnetroides
            Console.WriteLine($"Average Distance: {metrics.AverageDistance:F4}");
            // Indice bajo mejor, quiere decor que los clusters están bien separados
            Console.WriteLine($"Davies-Bouldin Index: {metrics.DaviesBouldinIndex:F4}");
            // Valores altos mejor. Solo si le indicamos el Label. No lo convierte en supervisado, por lo que devolverá NaN si no le das los resultados esperados
            Console.WriteLine($"Normalized Mutual Information: {metrics.NormalizedMutualInformation:F4}");

            var engine = mlContext.Model.CreatePredictionEngine<Point, PointPrediction>(model);

            Point[] pointsToPredict= [
                new Point { X = 1, Y = 1},
                new Point { X = 5, Y = 5},
                new Point { X = 1, Y = 10},
                new Point { X = 10, Y = 1},
            ];

            PointPrediction predictedPoint;

            foreach (var point in pointsToPredict)
                {
                    predictedPoint = engine.Predict(point);

                    var distancesFormatted = string.Join(", ", predictedPoint.Distances);
                    Console.WriteLine($"Punto ({point.X}, {point.Y}) => Cluster {predictedPoint.ClusterId}, Distances: [{distancesFormatted}]");
                }
        }
    }
}

public class Point
{
    [LoadColumn(1)]
    public float X { get; set; } = float.NaN;
    [LoadColumn(2)]
    public float Y { get; set; } = float.NaN;
}

public class PointPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }

    [ColumnName("Score")]
    public float[]? Distances { get; set; }
}