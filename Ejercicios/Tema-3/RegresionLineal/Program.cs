using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using RegresionLineal.Models;

namespace RegresionLineal
{
    public class EnergyPrediction
    {
        [ColumnName("Score")]
        public float Score { get; set; }
    }

    class Program
    {
        static void Main()
        {
            var mlContext = new MLContext(seed: 0);

            string dataPath = "./Data/energy_500.csv";

            var data = mlContext.Data.LoadFromTextFile<EnergyData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ','
            );

            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            var sdcaOptions = new SdcaRegressionTrainer.Options
            {
                LabelColumnName = nameof(EnergyData.EnergyKWh),
                FeatureColumnName = "Features",
                L1Regularization = 1e-7f,
                L2Regularization = 0.01f,
                BiasLearningRate = 0.1f,
                MaximumNumberOfIterations = 10,
                LossFunction = new SquaredLoss(),
                ConvergenceTolerance = 0.1f,
                ConvergenceCheckFrequency = 2,
                Shuffle = true,
                NumberOfThreads = 2
            };

            var pipeline =
                mlContext.Transforms.CopyColumns("Label", nameof(EnergyData.EnergyKWh))
                .Append(mlContext.Transforms.Concatenate("Features",
                    nameof(EnergyData.TempC),
                    nameof(EnergyData.HumidityPct),
                    nameof(EnergyData.PressureBar),
                    nameof(EnergyData.LoadPct),
                    nameof(EnergyData.Vibration)))
                .Append(mlContext.Regression.Trainers.Sdca(sdcaOptions));

            var model = pipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.Regression.Evaluate(predictions);

            PrintMetrics(metrics);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<EnergyData, EnergyPrediction>(model);

            var testCases = new[]
            {
                new EnergyData { TempC = 20, HumidityPct = 45, PressureBar = 1.010f, LoadPct = 35, Vibration = 1.80f },
                new EnergyData { TempC = 22, HumidityPct = 50, PressureBar = 1.020f, LoadPct = 68, Vibration = 2.80f },
                new EnergyData { TempC = 19, HumidityPct = 55, PressureBar = 1.035f, LoadPct = 95, Vibration = 3.90f },
                new EnergyData { TempC = 23.23f, HumidityPct = 41.3f, PressureBar = 1.040f, LoadPct = 82.6f, Vibration = 3.20f }
            };

            Console.WriteLine("\n===== PREDICCIONES =====");
            Console.WriteLine($"{"TempC",10} {"Hum%",10} {"Press",10} {"Load%",10} {"Vib",10} {"Pred KWh",10}");
            Console.WriteLine(new string('-', 70));

            foreach (var d in testCases)
            {
                var result = predictionEngine.Predict(d);

                Console.WriteLine($"{d.TempC,10:F1} {d.HumidityPct,10:F1} {d.PressureBar,10:F3} {d.LoadPct,10:F1} {d.Vibration,10:F2} {result.Score,10:F2}");
            }

            Console.WriteLine(new string('-', 70));
        }

        static void PrintMetrics(RegressionMetrics metrics)
        {
            Console.WriteLine("===== MÉTRICAS =====");
            Console.WriteLine($"MAE : {metrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
            Console.WriteLine($"MSE : {metrics.MeanSquaredError:F4}");
            Console.WriteLine($"R²  : {metrics.RSquared:F4}");
            Console.WriteLine($"loss: {metrics.LossFunction:F4}");
        }
    }
}