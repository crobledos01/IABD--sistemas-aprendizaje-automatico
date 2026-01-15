using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using DeteccionDeAnomalias.Models;

////////////////////////////////////
////// Contexto
////////////////////////////////////

var mlContext = new MLContext(seed: 1);

const string DatasetPath = "transacciones.csv";


////////////////////////////////////
////// Cargamos los datos
////////////////////////////////////

IDataView dataView = mlContext.Data.LoadFromTextFile<TranData>(
    path: DatasetPath,
    hasHeader: true,
    separatorChar: ',');

// Entreno con los datos "normales", hasta el 30. Esto lo hace Pablo porque en su dataset los primeros 30 datos son normales y los últimos 5 son los anomalos
// así entrena el modelo solo con los valores funcionales. En mi caso, los anómalos son USR5621 (TXN005-007), USR4532 (TXN019-024), USR9103 (TXN029-030), USR7156 (TXN041-045)
var trainingData = mlContext.Data.FilterRowsByColumn(
    dataView,
    nameof(TranData.Minute),
    upperBound: 30);


////////////////////////////////////
////// Pipeline de PROCESAMIENTO
////////////////////////////////////
var preprocessingPipeline =
    mlContext.Transforms.Concatenate("Features",
    [
        nameof(TranData.idTran),
        nameof(TranData.idUser),
        nameof(TranData.Date),
        nameof(TranData.Hour),
        nameof(TranData.Import),
        nameof(TranData.Country),
        nameof(TranData.Type),
        nameof(TranData.NumTrans),
        nameof(TranData.PayMethod),
    ])
    .Append(mlContext.Transforms.NormalizeMeanVariance("Features"));


////////////////////////////////////
////// Pipeline de DETECCIÓN ANOML.
////////////////////////////////////

var rPcaPipeline =
    preprocessingPipeline.Append(
        mlContext.AnomalyDetection.Trainers.RandomizedPca(
            featureColumnName: "Features",
            rank: 2
        )
    );



////////////////////////////////////
////// Creación del modelo
////// M = {PREPRO + RandomicedPCA}
////////////////////////////////////

// semi--supervisado
var model = rPcaPipeline.Fit(trainingData);

// no supervisado
// var model = rPcaPipeline.Fit(dataView);



////////////////////////////////////
////// Modelo anterior con diferentes threshold
////// M = {PREPRO + RandomicedPCA + Custom Threshold}
////////////////////////////////////

foreach (var t in new[] { 0.30f, 0.45f, 0.60f, 0.50f })
{
    var preproModel = model.Take(model.Count() - 1);
    var pca = (AnomalyPredictionTransformer<PcaModelParameters>)model.LastTransformer;

    // Cambia el umbral SOLO del último transformer
    var pcaWithThreshold = mlContext.AnomalyDetection.ChangeModelThreshold(
        pca,
        threshold: t);

    // Reconstruye la cadena sustituyendo el último transformer
    var thresholdedModel = model.Append(pcaWithThreshold);

    // Aplica el modelo con ese umbral
    var scored = thresholdedModel.Transform(dataView);

    var count = mlContext.Data
        .CreateEnumerable<TranData>(scored, reuseRowObject: false)
        .Count(r => r.PredictedLabel);

    Console.WriteLine($"Threshold {t:F2}: {count} anomalías detectadas");
}

////////////////////////////////////
////// Consumir el modelo
////////////////////////////////////
var predictions = model.Transform(dataView);


////////////////////////////////////
////// Medir el modelo
////////////////////////////////////
var metrics = mlContext.AnomalyDetection.Evaluate(
    data: predictions,
    labelColumnName: "Label",
    scoreColumnName: "Score",
    predictedLabelColumnName: "PredictedLabel",
    falsePositiveCount: 5);

Console.WriteLine($"AU ROC: {metrics.AreaUnderRocCurve:F4}");
Console.WriteLine($"DR at FP=5: {metrics.DetectionRateAtFalsePositiveCount:F4}");

// Creamos un enumerable, para poder visualizar
var results = mlContext.Data.CreateEnumerable<HvacPrediction>(
    predictions,
    reuseRowObject: false);

// Mostramos los datos con un formateo friendly
Console.WriteLine("IDTran\tIDUser\tDate\tHour\tImport\tCountry\tType\tNumTrans\tMethod\tAnomaly\tScore");
Console.WriteLine("-------------------------------------------------------------");

foreach (var r in results)
{
    Console.WriteLine(
        $"{r.idTran}\t{r.idUser:F1}\t{r.Date:F1}\t{r.Hour:F0}\t{r.Import:F1}\t{r.Country}\t{r.Type:F1}\t{r.NumTrans:F1}\t{r.PayMethod:F0}\t" +
        $"{(r.PredictedLabel ? "🔴" : "🟢")}\t{r.Score:F4}");
}

Console.WriteLine("-------------------------------------------------------------");
