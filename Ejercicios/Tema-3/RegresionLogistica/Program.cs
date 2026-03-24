using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using RegresionLogistica.Models;

namespace RegresionLogistica
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "./Data/system_500.csv";
            string resultsPath = "resultado.csv";
            Directory.CreateDirectory(resultsPath);

            var mlContext = new MLContext(seed: 1);

            IDataView data = mlContext.Data.LoadFromTextFile<SensorData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            var sdcaOptions = new SdcaLogisticRegressionBinaryTrainer.Options
            {
                LabelColumnName = nameof(SensorData.IsAnomaly),
                FeatureColumnName = "Features",
                L1Regularization = 0.001f,
                L2Regularization = 0.001f,
                MaximumNumberOfIterations = 100,
                ConvergenceTolerance = 0.01f,
                PositiveInstanceWeight = 1.2f,
                Shuffle = true,
                NumberOfThreads = Environment.ProcessorCount
            };

            Console.WriteLine("\n===== MODELO NO NORMALIZADO =====\n");
            var pipelineNoNorm = mlContext.Transforms.Concatenate("Features",
                        nameof(SensorData.TempC),
                        nameof(SensorData.HumPct),
                        nameof(SensorData.PowerW),
                        nameof(SensorData.DeltaT))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(sdcaOptions));

            var modelNoNorm = pipelineNoNorm.Fit(trainData);

            var predictionsNoNorm = modelNoNorm.Transform(testData);

            var metricsNoNorm = mlContext.BinaryClassification.Evaluate(
                data: predictionsNoNorm,
                labelColumnName: nameof(SensorData.IsAnomaly),
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel");

            PrintLogisticRegressionMetrics(metricsNoNorm);
            SaveLogisticRegressionMetrics(Path.Combine(resultsPath, "sin_normalizar"), metricsNoNorm);

            var logisticModelNoNorm =
                (CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>)modelNoNorm.LastTransformer.Model;
            ShowBiasAndWeights(logisticModelNoNorm);

            Console.WriteLine("\n===== MODELO NORMALIZADO =====\n");
            var pipelineNorm = mlContext.Transforms.Concatenate("Features",
                        nameof(SensorData.TempC),
                        nameof(SensorData.HumPct),
                        nameof(SensorData.PowerW),
                        nameof(SensorData.DeltaT))
                .Append(mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(sdcaOptions));

            var modelNorm = pipelineNorm.Fit(trainData);

            var predictionsNorm = modelNorm.Transform(testData);

            var metricsNorm = mlContext.BinaryClassification.Evaluate(
                data: predictionsNorm,
                labelColumnName: nameof(SensorData.IsAnomaly),
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel");

            PrintLogisticRegressionMetrics(metricsNorm);
            SaveLogisticRegressionMetrics(Path.Combine(resultsPath, "normalizado"), metricsNorm);

            var logisticModelNorm =
                (CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>)modelNorm.LastTransformer.Model;
            ShowBiasAndWeights(logisticModelNorm);

            // ============================
            //        PREDICCIONES
            // ============================

            var predictionEngine = mlContext.Model.CreatePredictionEngine<SensorData, ValuePrediction>(modelNorm);

            var testCases = new[]
            {
                new SensorData { TempC = 23f,  HumPct = 40f, PowerW = 510f, DeltaT = 5f },
                new SensorData { TempC = 37.5f,HumPct = 31f, PowerW = 770f, DeltaT = 17.5f },
                new SensorData { TempC = 24.5f,HumPct = 39f, PowerW = 525f, DeltaT = 6.5f },
                new SensorData { TempC = 40f,  HumPct = 30f, PowerW = 800f, DeltaT = 20f }
            };

            Console.WriteLine("\n===== PREDICCIONES =====");
            Console.WriteLine($"{"TempC",10} {"Hum%",10} {"PowerW",10} {"DeltaT",10} {"Clase",10} {"Prob.",10}");
            Console.WriteLine(new string('-', 65));

            for (int i = 0; i < testCases.Length; i++)
            {
                var result = predictionEngine.Predict(testCases[i]);
                var d = testCases[i];

                string clase = result.PredictedLabel ? "Anómalo" : "Normal";

                Console.WriteLine($"{d.TempC,10:F1} {d.HumPct,10:F1} {d.PowerW,10:F1} {d.DeltaT,10:F1} {clase,10} {result.Probability,10:P2}");
            }

            Console.WriteLine(new string('-', 65));
        }

        static void PrintLogisticRegressionMetrics(CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine("===== MÉTRICAS =====");
            Console.WriteLine($"Accuracy      : {metrics.Accuracy:F4}");
            Console.WriteLine($"AUC           : {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"F1Score       : {metrics.F1Score:F4}");
            Console.WriteLine($"Precision (N) : {metrics.NegativePrecision:F4}");
            Console.WriteLine($"Recall (N)    : {metrics.NegativeRecall:F4}");
            Console.WriteLine($"Precision (P) : {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Recall (P)    : {metrics.PositiveRecall:F4}");

            Console.WriteLine("----- matrix -----");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            Console.WriteLine("----- otras -----");
            Console.WriteLine($"loss (log loss)           : {metrics.LogLoss:F4}");
            Console.WriteLine($"loss reduction (log loss) : {metrics.LogLossReduction:F4}");
            Console.WriteLine($"Entropy                   : {metrics.Entropy:F4}");
            Console.WriteLine($"AUC P/R                   : {metrics.AreaUnderPrecisionRecallCurve:F4}");
        }

        static void SaveLogisticRegressionMetrics(string resultsPath, CalibratedBinaryClassificationMetrics metrics)
        {
            Directory.CreateDirectory(resultsPath);
            var metricsFile = Path.Combine(resultsPath, "metrics.txt");
            File.WriteAllText(metricsFile,
                $"Accuracy      : {metrics.Accuracy:F4}\n"
                + $"AUC           : {metrics.AreaUnderRocCurve:F4}\n"
                + $"F1Score       : {metrics.F1Score:F4}\n"
                + $"Precision (N) : {metrics.NegativePrecision:F4}\n"
                + $"Recall (N)    : {metrics.NegativeRecall:F4}\n"
                + $"Precision (P) : {metrics.PositivePrecision:F4}\n"
                + $"Recall (P)    : {metrics.PositiveRecall:F4}\n"
                + "----- matrix -----\n"
                + $"{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");
        }

        static void ShowBiasAndWeights(CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator> logisticModel)
        {
            Console.WriteLine("\n===== SESGO =====");
            Console.WriteLine($"Bias: {logisticModel.SubModel.Bias:F4}");

            Console.WriteLine("\n===== PESOS =====");
            var names = new[] { "TempC", "HumPct", "PowerW", "DeltaT" };
            var weights = logisticModel.SubModel.Weights.ToArray();
            for (int i = 0; i < weights.Length; i++)
            {
                Console.WriteLine($"{names[i]}: {weights[i]:F4}");
            }
        }
    }
    public class ValuePrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel { get; set; }
        [ColumnName("Clase")]
        public float Clase { get; set; }
        [ColumnName("Probability")]
        public float Probability { get; set; }
    }
}
