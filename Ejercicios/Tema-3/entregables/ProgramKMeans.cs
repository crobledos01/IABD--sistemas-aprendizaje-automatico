using System;
using System.Linq;
using System.Collections.Generic;
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

            // Carga los datos y los divide entre la parte de testeo y la de entrenamiento
            IDataView data = mlContext.Data.LoadFromTextFile<Clients>(path: fileInputPath, separatorChar: ',', hasHeader: true);
            var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Se crea una variable para comprobar cual es la métrica más positiva (la menor), otra para añadir el dato si
            // es nuevo K supera a la mejor hasta el momento (sirve para la K mínima también) y una última con el K máximo a comprobar
            double minorMetric = double.NaN;
            var bestK = 3;
            const int KTarget = 6;

            // Se crea un bucle que realiza las predicciones con cada K desde 3 hasta 6 para comprobar la cantidad de clústeres más óptima
            for (int k = bestK; k <= KTarget; k++)
            {
                Console.WriteLine();

                var pipelineK = mlContext.Transforms.Concatenate("Features", new[]
                {
                    "Edad", "NochesPorEstancia", "ViajaConNinos",
                    "GastoMedio", "DistanciaKm", "ReservasUltimoAnio"
                })
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: k));

                var modelK = pipelineK.Fit(splitData.TrainSet);

                var predictionsK = modelK.Transform(splitData.TestSet);

                ClusteringMetrics metricsK = mlContext.Clustering.Evaluate(
                    data: predictionsK,
                    scoreColumnName: "Score",
                    featureColumnName: "Features");

                // Console.WriteLine("At kluster = " + k);
                // Console.WriteLine($"Average Distance: {metricsK.AverageDistance:F4}");
                // Console.WriteLine($"Davies-Bouldin Index: {metricsK.DaviesBouldinIndex:F4}");

                var metricSumatory = metricsK.AverageDistance + metricsK.DaviesBouldinIndex;

                if (double.IsNaN(minorMetric) || metricSumatory < minorMetric)
                {
                    minorMetric = metricSumatory;
                    bestK = k;
                }
            }

            var finalPipeline = mlContext.Transforms.Concatenate("Features", new[] {
                "Edad", "NochesPorEstancia", "ViajaConNinos",
                "GastoMedio", "DistanciaKm", "ReservasUltimoAnio" })
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Clustering.Trainers.KMeans(numberOfClusters: bestK));

            var finalModel = finalPipeline.Fit(splitData.TrainSet);
            var finalPredictions = finalModel.Transform(splitData.TestSet);

            var finalMetrics = mlContext.Clustering.Evaluate(finalPredictions);

            Console.WriteLine("=== Métricas ===");
            Console.WriteLine($"Average Distance: {finalMetrics.AverageDistance:F4}");
            Console.WriteLine($"Davies-Bouldin Index: {finalMetrics.DaviesBouldinIndex:F4}");
            Console.WriteLine($"Normalized Mutual Information: {finalMetrics.NormalizedMutualInformation:F4}");

            var engine = mlContext.Model.CreatePredictionEngine<Clients, ClusterPrediction>(finalModel);

            // Convertimos los datos a lista
            var clientes = mlContext.Data
                .CreateEnumerable<Clients>(data, reuseRowObject: false)
                .ToList();

            var resultados = new List<(Clients Cliente, uint ClusterId)>();

            // Asigna los clientes a un cluster
            foreach (var c in clientes)
            {
                var pred = engine.Predict(c);
                resultados.Add((c, pred.ClusterId));
            }

            // Muestra las estadísticas de cada cluster
            foreach (var grp in resultados.GroupBy(r => r.ClusterId))
            {
                Console.WriteLine($"\n======= Cluster {grp.Key} =======");
                Console.WriteLine($" Edad media: {grp.Average(r => r.Cliente.Edad):F1}");
                Console.WriteLine($" Noches por estancia: {grp.Average(r => r.Cliente.NochesPorEstancia):F1}");
                Console.WriteLine($" % que viaja con niños: {grp.Average(r => r.Cliente.ViajaConNinos) * 100:F1}%");
                Console.WriteLine($" Gasto medio: {grp.Average(r => r.Cliente.GastoMedio):F0}€");
                Console.WriteLine($" Distancia media: {grp.Average(r => r.Cliente.DistanciaKm):F0} km");
                Console.WriteLine($" Reservas último año: {grp.Average(r => r.Cliente.ReservasUltimoAnio):F1}");
            }
        }
    }
}

public class Clients
{
    [LoadColumn(1)]
    public float Edad { get; set; }
    [LoadColumn(2)]
    public float NochesPorEstancia { get; set; }
    [LoadColumn(3)]
    public float ViajaConNinos { get; set; }
    [LoadColumn(4)]
    public float GastoMedio { get; set; }
    [LoadColumn(5)]
    public float DistanciaKm { get; set; }
    [LoadColumn(6)]
    public float ReservasUltimoAnio { get; set; }
}

public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint ClusterId { get; set; }

    [ColumnName("Score")]
    public float[] Distances { get; set; } = Array.Empty<float>();
}
