// Hacemos uso de implicit usings: https://devblogs.microsoft.com/dotnet/welcome-to-csharp-10/
using System.Text;
using System.Globalization;
using System.Runtime.CompilerServices;

class Ejemplo
{
    //////////////////////////////////////////////
    /// IMPUTAR DATOS
    //////////////////////////////////////////////

    [MethodImpl(MethodImplOptions.AggressiveInlining)] // Con esto le sugerimos al compiler que haga inline.
    static double Mean(IEnumerable<double> values) => values.Average();

    static double Median(IEnumerable<double> values)
    {
        var orderedData = values.OrderBy(v => v).ToArray();
        int dataLenght = orderedData.Length;

        return (dataLenght % 2 == 1) ?
            orderedData[dataLenght / 2] :
            (orderedData[dataLenght / 2 - 1] + orderedData[dataLenght / 2]) / 2.0;
    }

    static (double value, int count) Mode(IEnumerable<double> values, int decimals = 2)
    {
        return values
            .Select(v => Math.Round(v, decimals))
            .GroupBy(v => v)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key)
            .Select(g => (value: g.Key, count: g.Count()))
            .First();
    }

    static string ModeCategorical(IEnumerable<string> values)
    {
        return values
            .Where(s => !string.IsNullOrWhiteSpace(s))
            .GroupBy(s => s)
            .OrderByDescending(g => g.Count())
            .ThenBy(g => g.Key)
            .Select(g => g.Key)
            .FirstOrDefault() ?? string.Empty;
    }


    //////////////////////////////////////////////
    /// ESCALAR DATOS NUMERICOS
    //////////////////////////////////////////////

    // Min-Max [0,1]
    static double[] MinMax(double[] values)
    {
        double min = values.Min();
        double max = values.Max();
        if (max == min)
        {
            return values.Select(_ => 0.0).ToArray();
        }

        return values.Select(v => (v - min) / (max - min)).ToArray();
    }

    // Z-Score
    static double[] ZScore(double[] values)
    {
        double mean = Mean(values);

        // Sum of squares
        // Sumatorio (valor - media)^2
        // https://en.wikipedia.org/wiki/Sum_of_squares
        // https://en.wikipedia.org/wiki/Euclidean_distance
        double sumOfSquares = values.Select(v => Math.Pow(v - mean, 2)).Sum();

        double standardDeviation = Math.Sqrt(sumOfSquares / values.Length);
        if (standardDeviation == 0)
        {
            return values.Select(_ => 0.0).ToArray();
        }

        return values.Select(v => (v - mean) / standardDeviation).ToArray();
    }


    //////////////////////////////////////////////
    /// CODIFICAR VARIABLES CATEGORICAS
    //////////////////////////////////////////////

    // Label Encoder
    static Dictionary<string, int> CreateLabelEncoder(IEnumerable<string> values)
    {
        var encoder = new Dictionary<string, int>();
        int label = 0;
        foreach (var value in values)
        {
            var key = value ?? string.Empty;
            if (!encoder.ContainsKey(key))
            {
                encoder[key] = label;
                label++;
            }
        }

        return encoder;
    }

    // Label Encoding
    static (string header, int[] vector, Dictionary<string, int> encoding) LabelEncoding(string[] values, string colName)
    {
        var encoding = CreateLabelEncoder(values);
        var header = $"{colName}_Label";

        int n = values.Length;
        var vector = new int[n];

        for (int i = 0; i < n; i++)
        {
            vector[i] = encoding[values[i]];
        }

        return (header, vector, encoding);
    }

    // One-Hot Encoding
    static (string[] headers, int[][] matrix, Dictionary<string, int> encoding) OneHotEncoding(string[] values, string colName)
    {
        var encoding = CreateLabelEncoder(values);

        var categories = encoding.OrderBy(kv => kv.Value).Select(kv => kv.Key).ToArray();
        var headers = categories.Select(c => $"{colName}_{c}").ToArray();

        int n = values.Length;
        int k = encoding.Count;
        var matrix = new int[n][];

        for (int i = 0; i < n; i++)
        {
            matrix[i] = new int[k];

            int col = encoding[values[i]];
            matrix[i][col] = 1;
        }

        return (headers, matrix, encoding);
    }


    //////////////////////////////////////////////
    /// FICHEROS
    //////////////////////////////////////////////

    static List<string[]> ReadCsv(string path)
    {
        var lines = File.ReadAllLines(path, Encoding.UTF8)
                        .Where(l => !string.IsNullOrWhiteSpace(l))
                        .ToList();

        return [.. lines.Select(l => l.Split(','))];
        // spread syntax: [.. ] --> https://learn.microsoft.com/en-us/dotnet/csharp/whats-new/csharp-12#collection-expressions
    }

    static void WriteCsv(string path, List<string[]> outRows)
    {
        using var sw = new StreamWriter(path, false, new UTF8Encoding(false));
        foreach (var row in outRows)
        {
            sw.WriteLine(string.Join(",", row.Select(Safe)));
        }
    }

    // Un poco mas de informacion en:
    // https://learn.microsoft.com/en-us/dotnet/standard/garbage-collection/using-objects
    // https://learn.microsoft.com/en-us/dotnet/api/system.idisposable?view=net-9.0
    // https://stackoverflow.com/a/48530672/6307750

    // Equivalente a:
    /*
        using (var sw = new StreamWriter(path, false, new UTF8Encoding(false)))
        {
            foreach (var row in outRows)
            {
                sw.WriteLine(string.Join(",", row.Select(Safe)));
            }
        }
    */

    // Equivalente a:
    /*
        var sw = new StreamWriter(path, false, new UTF8Encoding(false));
        try
        {
            foreach (var row in outRows)
            {
                sw.WriteLine(string.Join(",", row.Select(Safe)));
            }
        }
        finally
        {
            sw.Dispose(); // cierra y libera el stream del fichero
        }
    */


    //////////////////////////////////////////////
    /// UTILIDADES
    //////////////////////////////////////////////

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static string Safe(string value) => value?.Trim() ?? string.Empty;

    static double? ToNullableDouble(string value)
    {
        value = Safe(value);
        if (string.IsNullOrEmpty(value) || value.Equals("NaN", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        if (double.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out double v))
        {
            return v;
        }

        if (double.TryParse(value, NumberStyles.Any, new CultureInfo("es-ES"), out v))
        {
            return v;
        }

        return null;
    }

    /// <summary>
    /// Converts a double value to a string using the invariant culture
    /// and up to six decimal places, removing trailing zeros.
    /// </summary>
    /// <param name="v">The double value to convert.</param>
    /// <returns>A string representation of the number with up to six decimal digits.</returns>
    static string StringToDouble(double v) => v.ToString("0.######", CultureInfo.InvariantCulture);


    //////////////////////////////////////////////
    /// PROGRAMA PRINCIPAL
    //////////////////////////////////////////////

    public static void EjecutarEjemplo()
    {
        // Archivo de entrada y salida
        const string fileInputPath = "data.csv";
        const string fileOutputPath = "data_preprocessed.csv";


        //////////////////////////////////////////////
        /// 1º LEEMOS FICHERO
        //////////////////////////////////////////////

        var rows = ReadCsv(fileInputPath);
        if (rows.Count < 2)
        {
            Console.WriteLine("CSV vacio o sin datos.");
            return;
        }


        //////////////////////////////////////////////
        // 2º Separamso cabecera de datos
        //////////////////////////////////////////////

        var header = rows[0];
        var data = rows.Skip(1).ToList(); // Skip 1 porque es el header.


        //////////////////////////////////////////////
        // 3º Mapeamos Nombre columna con indice de la columna.
        // Nos sera de utilidad mas adelante.
        // Ej. {"ID", 0}, {"Edad", 1}, ..., {"Tiempo_Empleo", 7}.
        //////////////////////////////////////////////

        var colIndexMap = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < header.Length; i++)
        {
            colIndexMap[header[i]] = i;
        }


        //////////////////////////////////////////////
        // ESTE PASO ES OPCIONAL!!
        //////////////////////////////////////////////
        /// Comprobacion de las columnas. ¿Son las esperadas?
        //////////////////////////////////////////////

        // Nombres esperados: ID,Edad,Genero,Ingresos_Mensuales,Gastos_Anuales,Educacion,Calificacion_Credito,Tiempo_Empleo
        string[] headerColumnNames = [
            "ID", "Edad", "Genero", "Ingresos_Mensuales", "Gastos_Anuales", "Educacion", "Calificacion_Credito", "Tiempo_Empleo",
        ];

        foreach (var col in headerColumnNames)
        {
            if (!colIndexMap.ContainsKey(col))
            {
                throw new Exception($"Falta columna: {col}");
            }
        }


        //////////////////////////////////////////////
        // 4º Imputamos valores restantes
        //////////////////////////////////////////////

        // Numericas
        string[] numericColumnNames = [
            "Edad", "Ingresos_Mensuales", "Gastos_Anuales", "Calificacion_Credito", "Tiempo_Empleo",
        ];

        foreach (var col in numericColumnNames)
        {
            int colIndex = colIndexMap[col];
            var valuesInsideCurrentCol = data.Select(row => ToNullableDouble(row[colIndex])).ToArray();

            // Obtenemos la media para imputar los valores restantes.
            double mean = Mean(valuesInsideCurrentCol.Where(v => v.HasValue).Select(v => v!.Value));

            for (int i = 0; i < data.Count; i++)
            {
                // Si la columna no tiene valor
                if (!valuesInsideCurrentCol[i].HasValue)
                {
                    data[i][colIndex] = StringToDouble(mean); // Le asignamos la media que hemos calculado.
                }
            }
        }

        // Categoricas
        string[] categoricalColumnNames = [
            "Genero", "Educacion",
        ];

        foreach (var col in categoricalColumnNames)
        {
            int colIndex = colIndexMap[col];
            var values = data.Select(row => Safe(row[colIndex])).ToArray();

            // Obtenemos la moda para imputar los valores restantes.
            string mode = ModeCategorical(values);

            // Recorremos todos los datos (las filas)
            for (int i = 0; i < data.Count; i++)
            {
                // Si la columna es null, vacia, or que solo tiene white-space caracteres
                // O
                // Si tiene el valor NaN
                if (string.IsNullOrWhiteSpace(values[i]) || values[i].Equals("NaN", StringComparison.OrdinalIgnoreCase))
                {
                    data[i][colIndex] = mode; // Le asignamos la moda que hemos calculado.
                }
            }
        }


        //////////////////////////////////////////////
        // 5º Escalado de variables numericas
        //////////////////////////////////////////////

        var scaledCols = new Dictionary<string, double[]>();
        foreach (var col in numericColumnNames)
        {
            int colIndex = colIndexMap[col];
            var values = data.Select(row => ToNullableDouble(row[colIndex])!.Value).ToArray();

            // En este caso calculo los dos, pero debemos elegir uno u otro.
            scaledCols[col + "_MinMax"] = MinMax(values);
            scaledCols[col + "_ZScore"] = ZScore(values);
        }


        //////////////////////////////////////////////
        // 6º Codificacion de variables categoricas
        //////////////////////////////////////////////

        // Genero one-hot
        string[] generoVals = data.Select(row => Safe(row[colIndexMap["Genero"]])).ToArray();
        var (genHeaders, genOheMatrix, genEncoder) = OneHotEncoding(generoVals, "Genero");

        // Educacion one-hot
        string[] eduVals = data.Select(row => Safe(row[colIndexMap["Educacion"]])).ToArray();
        var (eduHeaders, eduOheMatrix, eduEncoder) = OneHotEncoding(eduVals, "Educacion");


        //////////////////////////////////////////////
        /// 7º GENERACION DE NUEVAS CARACTERiSTICAS
        //////////////////////////////////////////////

        // Ratio_Deuda = Gastos_Anuales / (Ingresos_Mensuales * 12)
        double[] ratio = data.Select(row =>
        {
            double gastos = ToNullableDouble(row[colIndexMap["Gastos_Anuales"]])!.Value;
            double ingresosM = ToNullableDouble(row[colIndexMap["Ingresos_Mensuales"]])!.Value;

            double denominator = ingresosM * 12.0;
            return denominator == 0 ? 0.0 : gastos / denominator;
        }).ToArray();


        //////////////////////////////////////////////
        /// 8º FORMATEAR LA SALIDA
        //////////////////////////////////////////////

        // Cabecera base: ID + numericas imputadas
        var outHeader = new List<string>();
        outHeader.Add("ID");
        outHeader.AddRange(numericColumnNames); // valores imputados
        outHeader.AddRange(genHeaders); // OHE genero
        outHeader.AddRange(eduHeaders); // OHE educacion
        outHeader.Add("Ratio_Deuda"); // nueva caracteristica

        // añadir escaladas (min-max y z-score)
        outHeader.AddRange(scaledCols.Keys);

        var outRows = new List<string[]>
        {
            outHeader.ToArray() // Con la cabecera
        };

        for (int i = 0; i < data.Count; i++)
        {
            var row = new List<string>();
            // ID
            row.Add(Safe(data[i][colIndexMap["ID"]]));

            // numericas imputadas
            foreach (var c in numericColumnNames)
            {
                row.Add(Safe(data[i][colIndexMap[c]]));
            }

            // OHE genero
            for (int k = 0; k < genHeaders.Length; k++)
            {
                row.Add(genOheMatrix[i][k].ToString());
            }

            // OHE educacion
            for (int k = 0; k < eduHeaders.Length; k++)
            {
                row.Add(eduOheMatrix[i][k].ToString());
            }

            // ratio
            row.Add(StringToDouble(ratio[i]));

            // escaladas
            foreach (var kv in scaledCols)
            {
                row.Add(StringToDouble(kv.Value[i]));
            }

            outRows.Add(row.ToArray());
        }


        //////////////////////////////////////////////
        /// 9º ESCRIBIMOS EL FICHERO DE SALIDA
        //////////////////////////////////////////////

        WriteCsv(fileOutputPath, outRows);

        Console.WriteLine($"OK -> {fileOutputPath}");
    }
}


