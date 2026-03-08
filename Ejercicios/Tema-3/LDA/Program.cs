using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

var mlContext = new MLContext(seed: 1);

var samples = new List<TextData>()
{
    // ==================================================
    // REAL MADRID
    // ==================================================
    new() { Text = "Victoria del Real Madrid con doblete de Vinicius en el Bernabéu" },
    new() { Text = "El Real Madrid ficha a un nuevo portero para la próxima temporada" },
    new() { Text = "Bellingham anota y el Madrid sigue líder de la Liga" },
    new() { Text = "El Real Madrid avanza a semifinales de la Champions League" },
    new() { Text = "Aficionados celebran el título del Madrid en Cibeles" },

    // ==================================================
    // FC BARCELONA
    // ==================================================
    new() { Text = "El FC Barcelona remonta un partido épico en el Camp Nou" },
    new() { Text = "Flick elogia a Pedri tras la victoria del FC Barcelona ante el Sevilla" },
    new() { Text = "Lewandowski marca dos goles en la victoria del FC Barcelona" },
    new() { Text = "El FC Barcelona presenta su nueva camiseta para la temporada que viene" },
    new() { Text = "Los aficionados del FC Barcelona cantan el himno en el estadio" },

    // ==================================================
    // ATLÉTICO DE MADRID
    // ==================================================
    new() { Text = "El Atlético gana en el Metropolitano con doblete de Griezmann" },
    new() { Text = "Simeone destaca la intensidad del Atlético tras vencer al Betis" },
    new() { Text = "Partido tenso entre Atlético y Real Sociedad con muchas tarjetas" },
    new() { Text = "Morata vuelve al gol y da la victoria al Atlético de Madrid" },
    new() { Text = "Ambiente espectacular de los aficionados en el Metropolitano" },

    // ==================================================
    // CIUDAD DE MADRID
    // ==================================================
    new() { Text = "Madrid celebra el carnaval con desfiles en el centro" },
    new() { Text = "Los museos de Madrid registran récord de visitantes este fin de semana" },
    new() { Text = "Nueva línea de metro conectará el aeropuerto con el centro de Madrid" },
    new() { Text = "El Retiro y la Gran Vía llenos de turistas en Semana Santa" },
    new() { Text = "Temperaturas suaves en Madrid con cielos despejados" },

    // ==================================================
    // CIUDAD DE BARCELONA
    // ==================================================
    new() { Text = "Barcelona acoge un gran festival de música internacional" },
    new() { Text = "Turistas visitan la Sagrada Familia y las playas de Barcelona" },
    new() { Text = "El Ayuntamiento de Barcelona mejora el transporte público" },
    new() { Text = "Gràcia celebra sus fiestas populares con calles decoradas" },
    new() { Text = "Barcelona disfruta de un clima soleado durante el fin de semana" },
};

var dataview = mlContext.Data.LoadFromEnumerable(samples);

var domainStop = new[]
{
    "la", "el", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "con", "para", "por", "sobre",
    "este", "esta", "esa", "aquel", "aquella",
    "mi", "tu", "su", "nuestro", "vuestro"
};

var pipeline =
    // Limpieza
    mlContext.Transforms.Text.NormalizeText(
        outputColumnName: "NormText",
        inputColumnName: "Text",
        caseMode: TextNormalizingEstimator.CaseMode.Lower,
        keepDiacritics: false,
        keepNumbers: true,
        keepPunctuations: false)

    // Tokenización
    .Append(mlContext.Transforms.Text.TokenizeIntoWords(
        outputColumnName: "Words",
        inputColumnName: "NormText"))

    // Stopwords de idioma y dominio
    .Append(mlContext.Transforms.Text.RemoveDefaultStopWords(
        outputColumnName: "WordsNoStop",
        inputColumnName: "Words",
        language: StopWordsRemovingEstimator.Language.Spanish))
    .Append(mlContext.Transforms.Text.RemoveStopWords(
        outputColumnName: "WordsNoStop",
        inputColumnName: "WordsNoStop",
        stopwords: domainStop))

    // Mapear palabras
    .Append(mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: "WordKeys",
        inputColumnName: "WordsNoStop"))

    // N-Grams
    .Append(mlContext.Transforms.Text.ProduceNgrams(
        outputColumnName: "Ngrams",
        inputColumnName: "WordKeys",
        ngramLength: 2,
        useAllLengths: false,
        weighting: NgramExtractingEstimator.WeightingCriteria.Tf))

    // Modelaje del LDA
    .Append(mlContext.Transforms.Text.LatentDirichletAllocation(
        outputColumnName: "Features",
        inputColumnName: "Ngrams",
        numberOfTopics: 5,
        alphaSum: 0.1f,
        beta: 0.1f,
        samplingStepCount: 4,
        maximumNumberOfIterations: 5000,
        maximumTokenCountPerDocument: 7,
        numberOfSummaryTermsPerTopic: 5));

// ==================================================
// Entrenar modelo
// ==================================================
var transformer = pipeline.Fit(dataview);

// ==================================================
// Motor de predicción
// ==================================================
var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(transformer);

Console.WriteLine("==============================");
Console.WriteLine("Topic1  Topic2  Topic3  Topic4  Topic5");

// Ejemplo de predicciones:
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Vinicius marca dos goles en el Bernabéu" }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Flick destaca el trabajo del Fútbol Club Barcelona ante el Real Madrid Club de Fútbol" }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Griezmann y Koke anotan para el Atlético de Madrid en el Metropolitano" }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Madrid inaugura una nueva línea de metro" }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "Festival de verano en las playas de Barcelona" }));
PrintLdaFeatures(predictionEngine.Predict(new() { Text = "El Atletico de Madrid se mide ante el Fútbol Club Barcelona en Madrid en su patido de liga" }));


// ==================================================
// Función auxiliar
// ==================================================
static void PrintLdaFeatures(TransformedTextData prediction)
{
    for (int i = 0; i < prediction.Features.Length; i++)
    {
        Console.Write($"{prediction.Features[i]:F4}  ");
    }
    Console.WriteLine($"{prediction.Text}");
}

// ==================================================
// Clases para datos
// ==================================================
class TextData
{
    public string Text { get; set; }
}

class TransformedTextData : TextData
{
    public float[] Features { get; set; }
}
