using System;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ClusteringKMeans
{
    class Program
    {
        static void Main()
        {
            const string fileInputPath = "clientes_casarural.csv";

            var mlContext = new MLContext();

            IDataView data = mlContext.Data.LoadFromTextFile<Clients>(path: fileInputPath, separatorChar: ',', hasHeader: true);
            var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var minorMetrick = float.NaN;
            var bestK = 2;
            const int KTarget = 6;

            for (int k = bestK; k <= KTarget; k++)
            {
                Console.WriteLine("");

                var pipelineK = mlContext.Transforms.Concatenate(outputColumnName: "Features", inputColumnNames: ["IdCliente" ,"Edad" ,"NochesPorEstancia" ,"ViajaConNinos" , "GastoMedio", "DistanciaKm", "ReservasUltimoAnio"])
                        .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: k));

                var modelK = pipelineK.Fit(splitData.TrainSet);

                var predictionsK = modelK.Transform(splitData.TestSet);

                ClusteringMetrics metricsK = mlContext.Clustering.Evaluate(
                    data: predictionsK,
                    scoreColumnName: "Score",
                    featureColumnName:"Features");

                Console.WriteLine("At kluster = " + k);
                Console.WriteLine($"Average Distance: {metricsK.AverageDistance:F4}");
                Console.WriteLine($"Davies-Bouldin Index: {metricsK.DaviesBouldinIndex:F4}");

                var metricksSumatory = metricsK.AverageDistance + metricsK.DaviesBouldinIndex;

                if (minorMetrick == float.NaN || metricksSumatory < minorMetrick)
                {
                    bestK = k;
                }
            }
        }
    }
}

public class Clients
{
    [LoadColumn(1)]
    public float IdCliente { get; set; } = float.NaN;
    [LoadColumn(2)]
    public float Edad { get; set; } = float.NaN;
    [LoadColumn(3)]
    public float NochesPorEstancia { get; set; } = float.NaN;
    [LoadColumn(4)]
    public float ViajaConNinos { get; set; } = float.NaN;
    [LoadColumn(5)]
    public float GastoMedio { get; set; } = float.NaN;
    [LoadColumn(6)]
    public float DistanciaKm { get; set; } = float.NaN;
    [LoadColumn(7)]
    public float ReservasUltimoAnio { get; set; } = float.NaN;
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }

    [ColumnName("Score")]
    public float[] Distances { get; set; } = Array.Empty<float>();
}
