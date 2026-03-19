using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

var mlContext = new MLContext(seed: 1);

var samples = new List<TextData>()
{
    // ==================================================
    // TEMA 1: FÚTBOL
    // ==================================================
    new() { Text = "En el partido de fútbol del fin de semana, el equipo local dominó la posesión del balón, presionó muy arriba y generó numerosas ocasiones de gol, mientras la afición llenaba el estadio con cánticos y banderas, celebrando cada jugada de ataque y demostrando una pasión inagotable por los colores de su club." },
    new() { Text = "El entrenador del equipo explicó en rueda de prensa que la estrategia se basó en un sistema de presión intensa, líneas muy juntas y transiciones rápidas, destacando el trabajo defensivo de los centrocampistas y la capacidad de los delanteros para aprovechar cada espacio que dejaba la defensa rival durante el partido de fútbol." },
    new() { Text = "La liga de fútbol entra en una fase decisiva, con varios equipos peleando por el título, otros luchando por puestos de competiciones continentales y algunos tratando de evitar el descenso, mientras la prensa analiza al detalle las estadísticas, los esquemas tácticos y el rendimiento individual de las principales figuras." },
    new() { Text = "Durante el clásico más esperado de la temporada, el ambiente en las gradas fue espectacular, con tifos, bengalas de colores y cánticos que no cesaron, mientras los jugadores mostraban un alto nivel de intensidad, entrando fuerte a cada balón dividido y disputando cada minuto como si fuera el último." },
    new() { Text = "La cantera del club de fútbol sigue produciendo jóvenes talentos que destacan por su técnica y visión de juego, y muchos de ellos debutan pronto en el primer equipo, combinando la ilusión de sus primeros minutos con la responsabilidad de representar a una institución histórica frente a miles de aficionados." },
    new() { Text = "Los medios deportivos dedicaron amplias portadas al derbi de la ciudad, un partido de fútbol lleno de emoción, polémicas arbitrales y goles en los últimos instantes, que dejó a una afición eufórica y a la otra frustrada, alimentando la rivalidad y las conversaciones en bares y redes sociales durante días." },
    new() { Text = "La selección nacional de fútbol presentó su lista de convocados para el próximo torneo internacional, combinando jugadores experimentados con jóvenes promesas, mientras el seleccionador explicaba la importancia del equilibrio entre defensa y ataque, así como la necesidad de un grupo unido y comprometido." },
    new() { Text = "La tecnología del VAR ha cambiado la forma de vivir los partidos de fútbol, introduciendo revisiones detalladas de jugadas polémicas, goles anulados y penaltis revisados, lo que genera debates constantes sobre la justicia de las decisiones y el ritmo del juego, tanto entre periodistas como entre aficionados." },
    new() { Text = "Los entrenamientos del equipo de fútbol incluyen sesiones tácticas en el campo, trabajo físico en el gimnasio y análisis de vídeo, donde el cuerpo técnico revisa las acciones más importantes de los partidos anteriores para corregir errores, reforzar automatismos y preparar mejor el siguiente compromiso." },
    new() { Text = "La final de la copa de fútbol se disputó en un estadio neutral, con aficionados de ambos equipos viajando desde distintas regiones, creando un ambiente festivo en la ciudad anfitriona, ocupando plazas, restaurantes y hoteles, y convirtiendo el encuentro en un verdadero evento social más allá de lo estrictamente deportivo." },

    // ==================================================
    // TEMA 2: POLÍTICA
    // ==================================================
    new() { Text = "El parlamento debatió durante horas una reforma política que pretende modificar el sistema electoral, con discursos intensos de los distintos partidos, protestas en las calles y una fuerte atención mediática sobre las posibles consecuencias que estos cambios podrían tener en la representación ciudadana." },
    new() { Text = "La campaña electoral se llenó de mítines, entrevistas y debates televisivos, donde los candidatos presentaban sus propuestas en materia de economía, sanidad y empleo, mientras los analistas políticos destacaban la importancia del voto indeciso y las redes sociales amplificaban cada gesto y declaración." },
    new() { Text = "El presidente del gobierno anunció un nuevo paquete de medidas políticas para afrontar la crisis económica, incluyendo ayudas directas a familias vulnerables, subvenciones a pequeñas empresas y planes de inversión en infraestructuras, generando reacciones diversas entre la oposición y los sindicatos." },
    new() { Text = "Un escándalo de corrupción política sacudió la escena pública cuando se filtraron documentos que implicaban a varios altos cargos, provocando dimisiones, investigaciones judiciales y un intenso debate sobre la transparencia, la ética y la necesidad de reforzar los mecanismos de control institucional." },
    new() { Text = "Las relaciones internacionales se vieron afectadas por un conflicto diplomático entre dos países vecinos, con declaraciones cruzadas, retirada de embajadores y negociaciones discretas para rebajar la tensión, mientras los ciudadanos seguían con preocupación las noticias sobre posibles sanciones y acuerdos." },
    new() { Text = "Un nuevo partido político emergió con fuerza en las encuestas, presentándose como una alternativa a las formaciones tradicionales, defendiendo un programa centrado en la regeneración democrática, la lucha contra la desigualdad y la protección del medio ambiente, especialmente entre los votantes jóvenes." },
    new() { Text = "El parlamento regional aprobó una ley polémica que afectaba a la educación y a la sanidad públicas, con manifestaciones multitudinarias en las calles, comunicados de asociaciones profesionales y un intenso debate político sobre el modelo de servicios esenciales que la sociedad necesita." },
    new() { Text = "Durante la cumbre internacional, los líderes políticos discutieron temas como el cambio climático, la seguridad energética y la cooperación en materia de migración, firmando declaraciones conjuntas que reflejaban compromisos generales pero dejando algunos puntos concretos para negociaciones futuras." },
    new() { Text = "La participación ciudadana en política aumentó gracias a nuevas plataformas digitales que facilitan la consulta de propuestas, la firma de iniciativas legislativas y el debate público, aunque también se detectaron riesgos de desinformación y polarización en determinados espacios en línea." },
    new() { Text = "En un histórico referéndum, la población fue llamada a votar sobre una cuestión clave para el futuro del país, con campañas a favor y en contra, debates familiares, movilización de organizaciones civiles y una jornada electoral seguida minuto a minuto por medios nacionales e internacionales." },

    // ==================================================
    // TEMA 3: RELIGIÓN
    // ==================================================
    new() { Text = "La comunidad religiosa celebró una ceremonia multitudinaria en el templo principal, con cánticos, oraciones y rituales que se transmiten de generación en generación, mientras los fieles expresaban su devoción y buscaban momentos de reflexión espiritual en medio de la vida cotidiana." },
    new() { Text = "Durante la festividad más importante del calendario religioso, las calles se llenaron de procesiones, imágenes sagradas, flores y velas, con familias que acudían juntas a los actos litúrgicos, mezclando tradición cultural, fe personal y un fuerte sentido de pertenencia a la comunidad." },
    new() { Text = "Un grupo de representantes de distintas religiones se reunió en un encuentro interreligioso para promover el diálogo, la tolerancia y el respeto mutuo, compartiendo experiencias sobre cómo sus creencias inspiran acciones solidarias y proyectos de ayuda a personas en situación de vulnerabilidad." },
    new() { Text = "En muchos hogares, la religión se vive a través de pequeñas prácticas diarias, como rezar antes de las comidas, encender velas en determinadas fechas o leer textos sagrados en familia, lo que contribuye a transmitir valores, relatos y costumbres a las nuevas generaciones." },
    new() { Text = "La figura de los líderes religiosos sigue siendo relevante para muchas personas, que acuden a ellos en busca de orientación espiritual, consuelo en momentos difíciles y acompañamiento en decisiones importantes, aunque también se debate sobre el papel de la religión en sociedades cada vez más secularizadas." },
    new() { Text = "Un antiguo monasterio se ha convertido en un lugar de retiro espiritual, donde los visitantes participan en jornadas de silencio, meditación y oración, alejándose temporalmente del ruido de la ciudad para reencontrarse con sus creencias y con un ritmo de vida más pausado." },
    new() { Text = "La enseñanza religiosa en escuelas y centros de estudio genera debates sobre la libertad de conciencia, la diversidad de creencias y la necesidad de ofrecer una educación que respete tanto a quienes profesan una fe concreta como a quienes se declaran no creyentes." },
    new() { Text = "Los textos sagrados de distintas tradiciones religiosas son objeto de estudio por parte de teólogos, historiadores y filólogos, que analizan su contexto, sus interpretaciones a lo largo del tiempo y su influencia en las normas éticas y culturales de diferentes sociedades." },
    new() { Text = "En ciertos barrios, las festividades religiosas se combinan con actividades culturales, conciertos de música espiritual, ferias de gastronomía típica y encuentros comunitarios, reforzando la identidad local y generando espacios de convivencia entre personas de orígenes diversos." },
    new() { Text = "Los medios de comunicación dedican espacios a cuestiones religiosas cuando se producen acontecimientos relevantes como la elección de un líder espiritual, la publicación de un documento doctrinal importante o la organización de grandes peregrinaciones que reúnen a miles de creyentes." },

    // ==================================================
    // TEMA 4: ATLETISMO
    // ==================================================
    new() { Text = "En la pista de atletismo, los velocistas se preparan para la final de los cien metros, ajustando sus salidas en los tacos, concentrándose en la respiración y visualizando la carrera, mientras el público espera con expectación el disparo que marcará el inicio de la prueba reina." },
    new() { Text = "El entrenamiento de fondo en atletismo incluye largas sesiones de carrera continua, series en distintas intensidades y ejercicios de fuerza, con el objetivo de mejorar la resistencia aeróbica, la velocidad y la capacidad de mantener un ritmo competitivo durante muchos minutos." },
    new() { Text = "En un gran campeonato de atletismo, se combinan disciplinas muy diferentes como saltos, lanzamientos y pruebas de pista, lo que convierte el estadio en un escenario dinámico donde cada atleta lucha por su marca personal, por una medalla o por la clasificación para eventos internacionales." },
    new() { Text = "La maratón de la ciudad reunió a miles de corredores aficionados y profesionales, que recorrieron calles emblemáticas animados por voluntarios y espectadores, mientras los organizadores controlaban el avituallamiento, la seguridad y los tiempos oficiales con sistemas electrónicos." },
    new() { Text = "En las categorías de base del atletismo, los jóvenes deportistas aprenden la técnica de carrera, la coordinación y la importancia del calentamiento, participando en competiciones escolares que fomentan hábitos de vida saludables y el gusto por el deporte desde edades tempranas." },
    new() { Text = "Los atletas de élite planifican su temporada con entrenadores y preparadores físicos, eligiendo cuidadosamente las competiciones de atletismo en las que participarán, adaptando las cargas de trabajo y cuidando la alimentación, el descanso y la prevención de lesiones." },
    new() { Text = "Las pruebas combinadas de atletismo, como el decatlón y el heptatlón, exigen una enorme versatilidad, ya que los competidores deben rendir en carreras, saltos y lanzamientos, gestionando su energía y su concentración durante dos días completos de competición." },
    new() { Text = "La evolución de las pistas sintéticas y del calzado deportivo ha contribuido a la mejora de las marcas en atletismo, aunque también se ha abierto un debate sobre los límites tecnológicos aceptables para mantener la igualdad de condiciones entre todos los participantes." },
    new() { Text = "En los campeonatos escolares de atletismo, padres y entrenadores animan desde la grada mientras los niños corren relevos, saltan longitud y lanzan peso, viviendo sus primeras experiencias competitivas y aprendiendo a aceptar tanto las victorias como las derrotas." },
    new() { Text = "Los jueces de atletismo cumplen un papel fundamental en las competiciones, verificando salidas, midiendo distancias, controlando los cambios de relevo y asegurándose de que todas las pruebas se desarrollan de acuerdo con el reglamento internacional vigente." },

    // ==================================================
    // TEMA 5: EDUCACIÓN
    // ==================================================
    new() { Text = "El sistema de educación pública busca garantizar que todos los niños tengan acceso a una escuela cercana, con profesorado cualificado, recursos didácticos suficientes y programas de apoyo para aquellos alumnos que presentan dificultades de aprendizaje o situaciones familiares complejas." },
    new() { Text = "En las aulas de educación primaria, los docentes combinan explicaciones teóricas con actividades prácticas, trabajo en grupo y proyectos creativos, promoviendo no solo la adquisición de contenidos sino también habilidades como la comunicación, la colaboración y el pensamiento crítico." },
    new() { Text = "La educación universitaria se enfrenta al reto de adaptar sus planes de estudio a las nuevas demandas del mercado laboral, incorporando competencias digitales, prácticas en empresas y metodologías activas que fomenten la autonomía y la capacidad de aprender a lo largo de la vida." },
    new() { Text = "Las nuevas tecnologías han transformado la educación, permitiendo clases en línea, plataformas de recursos, evaluaciones digitales y herramientas de comunicación entre profesores, estudiantes y familias, aunque también plantean desafíos relacionados con la brecha digital y la gestión del tiempo frente a las pantallas." },
    new() { Text = "La formación profesional ofrece itinerarios educativos orientados al empleo, con módulos prácticos, talleres especializados y estancias en empresas, lo que facilita que los estudiantes adquieran competencias técnicas concretas y puedan incorporarse al mercado laboral con mayor rapidez." },
    new() { Text = "Los programas de educación inclusiva trabajan para que los centros escolares sean espacios donde se respeten las diferencias individuales, se atienda a la diversidad funcional y cultural, y se generen entornos de aprendizaje en los que todos los alumnos se sientan valorados y apoyados." },
    new() { Text = "Los docentes participan en cursos de formación continua para actualizar sus conocimientos pedagógicos, aprender nuevas metodologías y mejorar su capacidad de gestión del aula, conscientes de que la educación evoluciona constantemente y que su papel es clave en ese proceso." },
    new() { Text = "Las familias desempeñan un papel fundamental en la educación de los hijos, colaborando con los centros escolares, supervisando tareas, fomentando hábitos de lectura y acompañando emocionalmente a los menores en las distintas etapas del sistema educativo." },
    new() { Text = "Los organismos internacionales publican informes periódicos sobre la calidad de la educación en distintos países, analizando indicadores como el abandono escolar, el rendimiento en competencias básicas y la inversión pública, con el objetivo de orientar políticas de mejora." },
    new() { Text = "Los proyectos de educación para adultos ofrecen oportunidades de retomar estudios, aprender nuevas habilidades profesionales o ampliar la cultura general, permitiendo que personas de distintas edades puedan mejorar sus perspectivas de empleo y su desarrollo personal." },
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
        useAllLengths: true,
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

// FÚTBOL
PrintLdaFeatures(predictionEngine.Predict(new()
{
    Text = "El equipo ganó el partido de fútbol con un gol en los últimos minutos."
}));

// POLÍTICA
PrintLdaFeatures(predictionEngine.Predict(new()
{
    Text = "El parlamento aprobó una nueva ley tras un intenso debate político."
}));

// RELIGIÓN
PrintLdaFeatures(predictionEngine.Predict(new()
{
    Text = "La comunidad se reunió en el templo para una ceremonia religiosa especial."
}));

// ATLETISMO
PrintLdaFeatures(predictionEngine.Predict(new()
{
    Text = "La atleta logró su mejor marca en la carrera de cien metros de atletismo."
}));

// EDUCACIÓN
PrintLdaFeatures(predictionEngine.Predict(new()
{
    Text = "El colegio implantó un nuevo programa de educación para mejorar el aprendizaje."
}));


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
