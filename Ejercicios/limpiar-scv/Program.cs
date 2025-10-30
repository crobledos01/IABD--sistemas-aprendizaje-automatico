using System;
using System.Text;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Linq;
using System.IO;

class Program
{
    //////////////////////////////////////////////
    /// IMPUTAR DATOS
    //////////////////////////////////////////////

    [MethodImpl(MethodImplOptions.AggressiveInlining)] // Con esto le sugerimos al compiler que haga inline.
    static double Mean(IEnumerable<double> values) => values.Average();

    static double Median(IEnumerable<double> values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    static (double value, int count) Mode(IEnumerable<double> values, int decimals = 2)
    {
        var round_values = values.Select(v => Math.Round(v, decimals));
        var groups = round_values.GroupBy(x => x).OrderByDescending(g => g.Count());
        var first = groups.FirstOrDefault();
        return first != null ? (first.Key, first.Count()) : (0.0, 0);
    }

    static string ModeCategorical(IEnumerable<string> values)
    {
        var valid = values.Where(v => !string.IsNullOrWhiteSpace(v));
        var groups = valid.GroupBy(x => x).OrderByDescending(g => g.Count());
        return groups.FirstOrDefault()?.Key ?? "";
    }

    //////////////////////////////////////////////
    /// ESCALAR DATOS NUMERICOS
    //////////////////////////////////////////////

    // Min-Max
    static double[] MinMax(double[] values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    // Z-Score
    static double[] ZScore(double[] values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    //////////////////////////////////////////////
    /// CODIFICAR VARIABLES CATEGORICAS
    //////////////////////////////////////////////

    // Label Encoder
    static Dictionary<string, int> CreateLabelEncoder(IEnumerable<string> values)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    // Label Encoding
    static (string header, int[] vector, Dictionary<string, int> encoding) LabelEncoding(string[] values, string colName)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    // One-Hot Encoding
    static (string[] headers, int[][] matrix, Dictionary<string, int> encoding) OneHotEncoding(string[] values, string colName)
    {
        // TODO: Implementar
        throw new NotImplementedException();
    }

    //////////////////////////////////////////////
    /// FICHEROS
    //////////////////////////////////////////////

    static List<string[]> ReadCsv(string path)
    {
        var rows = new List<string[]>();
        foreach (var line in System.IO.File.ReadLines(path))
        {
            rows.Add(line.Split(',')); 
        }
        return rows;
    }

    static void WriteCsv(string path, List<string[]> outRows)
    {
        using (var writer = new StreamWriter(path, false))
        {
            foreach (var row in outRows)
            {
                writer.WriteLine(string.Join(",", row));
            }
        }
    }

    //////////////////////////////////////////////
    /// UTILIDADES
    //////////////////////////////////////////////

    /// <summary>
    /// Returns a safe string by removing leading and trailing whitespace.
    /// If the value is null, it returns an empty string.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static string Safe(string value) => value?.Trim() ?? string.Empty;

    /// <summary>
    /// Converts a string to a nullable <see cref="double"/>.
    /// Returns <c>null</c> for empty input or the literal "NaN" (case-insensitive).
    /// Trims the input via <c>Safe</c>, then tries InvariantCulture and "es-ES".
    /// </summary>
    /// <param name="value">Input string that may contain a numeric value.</param>
    /// <returns>
    /// Parsed <see cref="double"/> if successful; otherwise <c>null</c>.
    /// </returns>
    /// <remarks>
    /// Parsing order: <see cref="CultureInfo.InvariantCulture"/> first, then <c>es-ES</c>.
    /// </remarks>
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

    static void Main()
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
        // TODO: Implementar


        //////////////////////////////////////////////
        // ESTE PASO ES OPCIONAL!!
        //////////////////////////////////////////////
        /// Comprobacion de las columnas. ¿Son las esperadas?
        //////////////////////////////////////////////

        // Nombres esperados: ID,Edad,Gnero,Ingresos_Mensuales,Gastos_Anuales,Educacion,Calificacion_Credito,Tiempo_Empleo
        string[] headerColumnNames = [
            "ID", "Edad", "Genero", "Ingresos_Mensuales", "Gastos_Anuales", "Educacion", "Calificacion_Credito", "Tiempo_Empleo",
        ];

        // TODO: Implementar


        //////////////////////////////////////////////
        // 4º Imputamos valores restantes
        //////////////////////////////////////////////

        // Numericas
        string[] numericColumnNames = [
            "Edad", "Ingresos_Mensuales", "Gastos_Anuales", "Calificacion_Credito", "Tiempo_Empleo",
        ];

        var num_modes = new Dictionary<string, double>();
        foreach (var column_name in numericColumnNames)
        {
            int id = colIndexMap[column_name];
            var nums = data.Select(row => ToNullableDouble(row[id])).Where(x => x.HasValue).Select(x => x.Value);
            var mode = Mode(nums).value;
            num_modes[column_name] = mode;
        }

        // Categoricas
        string[] categoricalColumnNames = [
            "Genero", "Educacion",
        ];

        var cat_modes = new Dictionary<string, string>();
        foreach (var column_name in categoricalColumnNames)
        {
            int idx = colIndexMap[column_name];
            var vals = data.Select(row => Safe(row[idx]));
            cat_modes[column_name] = ModeCategorical(vals);
        }


        //////////////////////////////////////////////
        // 5º Escalado de variables numericas
        //////////////////////////////////////////////

        var scaledCols = new Dictionary<string, double[]>();
        // TODO: Implementar


        //////////////////////////////////////////////
        // 6º Codificacion de variables categoricas
        //////////////////////////////////////////////

        // Genero one-hot
        string[] generoVals = data.Select(row => Safe(row[colIndexMap["Genero"]])).ToArray();
        // var (genHeaders, genOheMatrix, genEncoder) = // TODO: Implementar

        // Educacion one-hot
        string[] eduVals = data.Select(row => Safe(row[colIndexMap["Educacion"]])).ToArray();
        // var (eduHeaders, eduOheMatrix, eduEncoder) = // TODO: Implementar


        //////////////////////////////////////////////
        /// 7º GENERACION DE NUEVAS CARACTERISTICAS
        //////////////////////////////////////////////

        // TODO: Ratio_Deuda = Gastos_Anuales / (Ingresos_Mensuales * 12)
        double[] ratio = data.Select(row =>
        {
            // TODO: Implementar
            return 0.0;
        }).ToArray();


        //////////////////////////////////////////////
        /// 8º FORMATEAR LA SALIDA
        //////////////////////////////////////////////

        // Cabecera base: ID + numericas imputadas
        var outHeader = new List<string>();
        for (int j = 0; j < header.Length; j++)
        {
            outHeader.Add(header[j]);
        }

        var outRows = new List<string[]>
        {
            outHeader.ToArray() // Con la cabecera
        };

        for (int i = 0; i < data.Count; i++)
        {
            var row = new List<string>();
            for (int j = 0; j < header.Length; j++)
            {
                string column_name = header[j];
                string value = Safe(data[i][j]);
                if (num_modes.ContainsKey(column_name) && (string.IsNullOrEmpty(value) || value.Equals("NaN", StringComparison.OrdinalIgnoreCase)))
                    row.Add(StringToDouble(num_modes[column_name]));
                else if (cat_modes.ContainsKey(column_name) && string.IsNullOrEmpty(value))
                    row.Add(cat_modes[column_name]);
                else
                    row.Add(value);
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