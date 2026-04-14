using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

// ===============================
// DATOS DE ENTRENAMIENTO Y TEST
// ===============================

var trainingData = new List<DataPoint>
{
    new([1.0, 1.0], "Rojo"),
    new([1.5, 2.0], "Rojo"),
    new([1.2, 0.8], "Rojo"),
    new([0.9, 1.1], "Rojo"),
    new([1.3, 1.4], "Rojo"),
    new([1.7, 1.6], "Rojo"),
    new([0.8, 0.9], "Rojo"),
    new([1.4, 1.0], "Rojo"),
    new([2.2, 2.4], "Rojo"),

    new([3.0, 3.5], "Azul"),
    new([4.0, 4.5], "Azul"),
    new([3.2, 3.7], "Azul"),
    new([3.8, 4.2], "Azul"),
    new([4.1, 3.9], "Azul"),
    new([3.6, 4.4], "Azul"),
    new([2.9, 3.3], "Azul"),
    new([4.3, 4.1], "Azul"),
    new([2.5, 2.7], "Azul")
};

var testData = new List<DataPoint>
{
    new([1.4, 1.3], "Rojo"),
    new([3.5, 4.0], "Azul"),
    new([1.1, 0.9], "Rojo"),
    new([3.2, 3.8], "Azul"),
    new([2.3, 2.5], "Rojo"),
    new([2.6, 2.8], "Azul"),
    new([1.6, 1.5], "Rojo"),
    new([4.2, 4.0], "Azul")
};

int k = 4;
string positiveLabel = "Rojo";

// ===============================
// PARTE 1: KNN
// ===============================

Console.WriteLine("===== KNN =====");

var knn = new KNN(trainingData);

double[] nuevoPunto = [1.4, 1.3];
string prediccion = knn.Predict(nuevoPunto, k);

Console.WriteLine($"La clase predicha para [1.4, 1.3] es: {prediccion}");

var knnMetrics = Metrics.Evaluate(knn, testData, k, positiveLabel);

Console.WriteLine($"TP: {knnMetrics.TP}");
Console.WriteLine($"TN: {knnMetrics.TN}");
Console.WriteLine($"FP: {knnMetrics.FP}");
Console.WriteLine($"FN: {knnMetrics.FN}");

Console.WriteLine($"Accuracy: {knnMetrics.Accuracy():F2}");
Console.WriteLine($"Precision: {knnMetrics.Precision():F2}");
Console.WriteLine($"Recall: {knnMetrics.Recall():F2}");
Console.WriteLine($"F1 Score: {knnMetrics.F1():F2}");

Console.WriteLine();

// ===============================
// PARTE 2: SVM CON ML.NET
// ===============================

Console.WriteLine("===== SVM (ML.NET - LinearSvm) =====");

var mlContext = new MLContext(seed: 0);

var trainMl = trainingData.Select(p => new ModelInput
{
    Features = p.Features.Select(x => (float)x).ToArray(),
    Label = p.Label == positiveLabel
});

var testMl = testData.Select(p => new ModelInput
{
    Features = p.Features.Select(x => (float)x).ToArray(),
    Label = p.Label == positiveLabel
});

IDataView trainDataView = mlContext.Data.LoadFromEnumerable(trainMl);
IDataView testDataView = mlContext.Data.LoadFromEnumerable(testMl);

var pipeline = mlContext.BinaryClassification.Trainers.LinearSvm(
    labelColumnName: "Label",
    featureColumnName: "Features");

var model = pipeline.Fit(trainDataView);
var predictions = model.Transform(testDataView);

var svmMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(
    predictions,
    labelColumnName: "Label");

Console.WriteLine($"Accuracy: {svmMetrics.Accuracy:F2}");
Console.WriteLine($"F1 Score: {svmMetrics.F1Score:F2}");
Console.WriteLine($"Positive Precision: {svmMetrics.PositivePrecision:F2}");
Console.WriteLine($"Positive Recall: {svmMetrics.PositiveRecall:F2}");
Console.WriteLine($"Negative Precision: {svmMetrics.NegativePrecision:F2}");
Console.WriteLine($"Negative Recall: {svmMetrics.NegativeRecall:F2}");

if (svmMetrics.ConfusionMatrix != null)
{
    Console.WriteLine();
    Console.WriteLine("Matriz de confusión SVM:");
    Console.WriteLine(svmMetrics.ConfusionMatrix.GetFormattedConfusionTable());
}

// ===============================
// PARTE 3: COMPARACIÓN
// ===============================

Console.WriteLine();
Console.WriteLine("===== COMPARACIÓN FINAL =====");
Console.WriteLine($"KNN -> Accuracy: {knnMetrics.Accuracy():F2}, Precision: {knnMetrics.Precision():F2}, Recall: {knnMetrics.Recall():F2}, F1: {knnMetrics.F1():F2}");
Console.WriteLine($"SVM -> Accuracy: {svmMetrics.Accuracy:F2}, Precision: {svmMetrics.PositivePrecision:F2}, Recall: {svmMetrics.PositiveRecall:F2}, F1: {svmMetrics.F1Score:F2}");


// ===============================
// CLASES
// ===============================

public class DataPoint
{
    public double[] Features { get; set; }
    public string Label { get; set; }

    public DataPoint(double[] features, string label)
    {
        Features = features;
        Label = label;
    }
}

public class KNN
{
    private readonly List<DataPoint> trainingData;

    public KNN(List<DataPoint> trainingData)
    {
        this.trainingData = trainingData;
    }

    private static double EuclideanDistance(double[] a, double[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Los vectores deben tener la misma dimensión.");

        double sum = 0;

        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }

        return Math.Sqrt(sum);
    }

    public string Predict(double[] newPoint, int k)
    {
        if (k <= 0)
            throw new ArgumentException("k debe ser mayor que 0.");

        if (k > trainingData.Count)
            throw new ArgumentException("k no puede ser mayor que el número de ejemplos de entrenamiento.");

        var prediction = trainingData
            .Select(p => new
            {
                Label = p.Label,
                Distance = EuclideanDistance(p.Features, newPoint)
            })
            .OrderBy(x => x.Distance)
            .Take(k)
            .GroupBy(x => x.Label)
            .Select(g => new
            {
                Label = g.Key,
                Count = g.Count(),
                AvgDistance = g.Average(x => x.Distance)
            })
            .OrderByDescending(x => x.Count)
            .ThenBy(x => x.AvgDistance)
            .First();

        return prediction.Label;
    }
}

public class Metrics
{
    public int TP { get; set; }
    public int TN { get; set; }
    public int FP { get; set; }
    public int FN { get; set; }

    public double Accuracy()
    {
        int total = TP + TN + FP + FN;
        return total == 0 ? 0 : (double)(TP + TN) / total;
    }

    public double Precision()
    {
        return (TP + FP) == 0 ? 0 : (double)TP / (TP + FP);
    }

    public double Recall()
    {
        return (TP + FN) == 0 ? 0 : (double)TP / (TP + FN);
    }

    public double F1()
    {
        double p = Precision();
        double r = Recall();
        return (p + r) == 0 ? 0 : 2 * (p * r) / (p + r);
    }

    public static Metrics Evaluate(KNN knn, List<DataPoint> testData, int k, string positiveLabel)
    {
        var metrics = new Metrics();

        foreach (var point in testData)
        {
            string predicted = knn.Predict(point.Features, k);
            string actual = point.Label;

            if (actual == positiveLabel)
            {
                if (predicted == positiveLabel)
                    metrics.TP++;
                else
                    metrics.FN++;
            }
            else
            {
                if (predicted == positiveLabel)
                    metrics.FP++;
                else
                    metrics.TN++;
            }
        }

        return metrics;
    }
}

public class ModelInput
{
    [VectorType(2)]
    public float[] Features { get; set; }
    public bool Label { get; set; }
}

public class ModelPrediction
{
    public bool PredictedLabel { get; set; }
    public float Score { get; set; }
}