using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace ensambles
{
    public class Program
    {
        public static void Main(string[] args)
        {
            MLContext mlContext = new MLContext(seed: 0);
            
            // 1. Cargar datos
            string dataPath = "Data/customers.csv";

            IDataView data = mlContext.Data.LoadFromTextFile<CustomerData>(
                path: dataPath,
                hasHeader: true,
                separatorChar: ',');

            // 2. Dividir entre train/test
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            
            IDataView trainData = split.TrainSet;
            IDataView testData = split.TestSet;

            // 3. Pipeline común de preprocesamiento
            var preprocessingPipeline =
                mlContext.Transforms.Categorical.OneHotEncoding( outputColumnName: "RegionEncoded", inputColumnName: nameof(CustomerData.Region))
                .Append(mlContext.Transforms.Conversion.ConvertType("SubscribedNewsletterFloat", nameof(CustomerData.SubscribedNewsletter), DataKind.Single))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    nameof(CustomerData.Age),
                    nameof(CustomerData.Income),
                    nameof(CustomerData.PreviousPurchases),
                    nameof(CustomerData.WebVisits),
                    "SubscribedNewsletterFloat",
                    "RegionEncoded"));

            // 4. FASTFOREST
            var fastForestOptions = new FastForestBinaryTrainer.Options
            {
                NumberOfTrees = 150,
                NumberOfLeaves = 32,
                MinimumExampleCountPerLeaf = 10,
                FeatureFraction = 0.8f,
                FeatureFirstUsePenalty = 0.05f
            };

            var fastForestPipeline = preprocessingPipeline.Append(
                mlContext.BinaryClassification.Trainers.FastForest(fastForestOptions));

            var fastForestModel = fastForestPipeline.Fit(trainData);

            var fastForestPredictions = fastForestModel.Transform(testData);

            var fastForestMetrics = mlContext.BinaryClassification.EvaluateNonCalibrated(fastForestPredictions);

            // 5. LIGHTGBM
            var lightGbmOptions = new LightGbmBinaryTrainer.Options
            {
                NumberOfIterations = 100,
                NumberOfLeaves = 31,
                MinimumExampleCountPerLeaf = 10,
                LearningRate = 0.1f
            };

            var lightGbmPipeline = preprocessingPipeline.Append(
                mlContext.BinaryClassification.Trainers.LightGbm(lightGbmOptions));

            var lightGbmModel = lightGbmPipeline.Fit(trainData);

            var lightGbmPredictions = lightGbmModel.Transform(testData);

            var lightGbmMetrics = mlContext.BinaryClassification.Evaluate(lightGbmPredictions);

            // 6. Mostrar resultados
            Console.WriteLine("===== RESULTADOS FASTFOREST =====");
            PrintFastForestMetrics(fastForestMetrics);
            Console.WriteLine();
            Console.WriteLine("===== RESULTADOS LIGHTGBM =====");
            PrintLightGbmMetrics(lightGbmMetrics);

            // 7. Comparación
            // Estos dos métodos deben ir abajo del todo, después del punto 7
            static void PrintFastForestMetrics(BinaryClassificationMetrics metrics)
            {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine("Matriz de confusión:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            }

            static void PrintLightGbmMetrics(CalibratedBinaryClassificationMetrics metrics)
            {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F4}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F4}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F4}");
            Console.WriteLine($"Positive Precision: {metrics.PositivePrecision:F4}");
            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F4}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:F4}");
            Console.WriteLine("Matriz de confusión:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            }
    }
}
public class CustomerData
{
    [LoadColumn(0)]
    public float Age { get; set; }

    [LoadColumn(1)]
    public float Income { get; set; }

    [LoadColumn(2)]
    public float PreviousPurchases { get; set; }

    [LoadColumn(3)]
    public float WebVisits { get; set; }

    [LoadColumn(4)]
    public bool SubscribedNewsletter { get; set; }

    [LoadColumn(5)]
    public string? Region { get; set; }

    [LoadColumn(6)]
    public bool Label { get; set; }
    }
}