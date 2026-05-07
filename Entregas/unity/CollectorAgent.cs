using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class CollectorAgent : Agent
{
    [Header("References")]
    [SerializeField] private Transform rewardOrb;
    [SerializeField] private Transform penaltyOrb;

    [Header("Movement")]
    [SerializeField] private float moveSpeed = 5f;

    [Header("Arena")]
    [SerializeField] private float arenaLimit = 11.5f;
    [SerializeField] private float spawnHeight = 1.0f;

    [Header("Spawn")]
    [SerializeField] private float minSpawnDistance = 3f;
    [SerializeField] private int maxSpawnAttempts = 100;

    [Header("Rewards")]
    [SerializeField] private float rewardOrbReward = 1.0f;
    [SerializeField] private float penaltyOrbReward = -0.5f;
    [SerializeField] private float wallPenalty = -0.1f;
    [SerializeField] private float stepPenalty = -0.001f;

    [SerializeField] private int maxScorePerEpisode = 10;

    private Rigidbody rb;

    public int Score { get; private set; }

    protected override void Awake()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        Score = 0;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.localPosition = GetRandomPosition();
        rewardOrb.localPosition = GetRandomPositionAvoiding(transform.localPosition);
        penaltyOrb.localPosition = GetRandomPositionAvoiding(transform.localPosition, rewardOrb.localPosition);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        Vector3 agentPos = transform.localPosition;
        Vector3 rewardPos = rewardOrb.localPosition;
        Vector3 penaltyPos = penaltyOrb.localPosition;

        Vector3 directionToReward = rewardPos - agentPos;
        Vector3 directionToPenalty = penaltyPos - agentPos;

        // Posición del agente
        sensor.AddObservation(agentPos.x);
        sensor.AddObservation(agentPos.z);

        // Dirección normalizada hacia el objeto positivo
        sensor.AddObservation(directionToReward.normalized.x);
        sensor.AddObservation(directionToReward.normalized.z);

        // Distancia al objeto positivo
        sensor.AddObservation(directionToReward.magnitude);

        // Dirección normalizada hacia el objeto negativo
        sensor.AddObservation(directionToPenalty.normalized.x);
        sensor.AddObservation(directionToPenalty.normalized.z);

        // Distancia al objeto negativo
        sensor.AddObservation(directionToPenalty.magnitude);

        // Velocidad actual del agente
        sensor.AddObservation(rb.linearVelocity.x);
        sensor.AddObservation(rb.linearVelocity.z);
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        int action = actions.DiscreteActions[0];

        Vector3 movementDir = action switch
        {
            0 => Vector3.zero,
            1 => Vector3.forward,
            2 => Vector3.back,
            3 => Vector3.left,
            4 => Vector3.right,
            _ => Vector3.zero
        };

        Vector3 newPosition = rb.position + movementDir * (moveSpeed * Time.fixedDeltaTime);

        rb.MovePosition(newPosition);

        AddReward(stepPenalty);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        //...
    }

    private void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("RewardOrb"))
        {
            // Añadimos una recompensa positiva por coger objeto bueno
            AddReward(rewardOrbReward);
            
            Score++;
            rewardOrb.localPosition = GetRandomPositionAvoiding(transform.localPosition, penaltyOrb.localPosition);
            if (Score >= maxScorePerEpisode)
            {
                // Fin del episodio (siguiente punto)
                EndEpisode();
            }
        }
        if (other.CompareTag("PenaltyOrb"))
        {
            // Añadimos una recompensa negativa por coger objeto malo
            AddReward(penaltyOrbReward);

            penaltyOrb.localPosition = GetRandomPositionAvoiding(transform.localPosition, rewardOrb.localPosition);
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("ArenaWall"))
        {
            // Añadimos una recompensa negativa por chocar con una pared
            AddReward(wallPenalty);
        }
    }

    private Vector3 GetRandomPosition()
    {
        float x = Random.Range(-arenaLimit, arenaLimit);
        float z = Random.Range(-arenaLimit, arenaLimit);

        return new Vector3(x, spawnHeight, z);
    }

    private Vector3 GetRandomPositionAvoiding(params Vector3[] positionsToAvoid)
    {
        for (int i = 0; i < maxSpawnAttempts; i++)
        {
            Vector3 candidatePosition = GetRandomPosition();

            bool validPosition = true;

            foreach (Vector3 positionToAvoid in positionsToAvoid)
            {
                float distance = Vector3.Distance(
                    new Vector3(candidatePosition.x, 0f, candidatePosition.z),
                    new Vector3(positionToAvoid.x, 0f, positionToAvoid.z)
                );

                if (distance < minSpawnDistance)
                {
                    validPosition = false;
                    break;
                }
            }

            if (validPosition)
            {
                return candidatePosition;
            }
        }

        return GetRandomPosition();
    }
}
