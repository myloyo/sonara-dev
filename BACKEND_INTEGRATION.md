## Backend Integration

### Что Backend должен делать

1. **POST /upload** - получить файл, загрузить в MinIO, создать Job в БД, отправить в Kafka
2. **PUT /api/jobs/{id}** - ML Service будет вызывать после обработки
3. **Kafka Consumer** - слушать job.completed и job.failed, обновлять Job в БД
4. **GET /api/jobs/{id}/download** - скачать файл из MinIO

### Job Model

```csharp
public class Job
{
    public Guid Id { get; set; }
    public string Status { get; set; }        // Queued, Completed, Failed
    public string InputKey { get; set; }      // input/job-xxx.wav
    public string OutputKey { get; set; }     // output/job-xxx.wav (nullable)
    public string ErrorMessage { get; set; }  // error msg (nullable)
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public DateTime? StartedAt { get; set; }
    public DateTime? FinishedAt { get; set; }
}
```

### 1. Upload Endpoint

```csharp
[HttpPost("upload")]
public async Task<IActionResult> UploadAsync(IFormFile file)
{
    var jobId = Guid.NewGuid();
    var inputKey = $"input/job-{jobId}.wav";

    // Upload to MinIO
    using (FileStream fs = System.IO.File.OpenRead(file.FileName))
    {
        await minioClient.PutObjectAsync(new PutObjectArgs()
            .WithBucket("audio-files")
            .WithObject(inputKey)
            .WithStreamData(fs)
            .WithObjectSize(fs.Length)
            .WithContentType("audio/wav"));
    }

    // Create Job
    var job = new Job
    {
        Id = jobId,
        Status = "Queued",
        InputKey = inputKey,
        CreatedAt = DateTime.UtcNow,
        UpdatedAt = DateTime.UtcNow
    };
    _context.Jobs.Add(job);
    await _context.SaveChangesAsync();

    // Send to Kafka
    await kafkaProducer.ProduceAsync("job.prepared",
        new Message<string, string>
        {
            Key = jobId.ToString(),
            Value = JsonSerializer.Serialize(new { jobId, inputKey })
        });

    return Ok(job);
}
```

### 2. Update Job Endpoint

ML Service вызывает это после обработки:

```csharp
[HttpPut("{id:guid}")]
public async Task<IActionResult> UpdateJobAsync(Guid id, UpdateJobRequest req)
{
    var job = await _context.Jobs.FindAsync(id);
    if (job == null) return NotFound();

    job.Status = req.Status;
    if (req.OutputKey != null) job.OutputKey = req.OutputKey;
    if (req.ErrorMessage != null) job.ErrorMessage = req.ErrorMessage;
    job.FinishedAt = DateTime.UtcNow;
    job.UpdatedAt = DateTime.UtcNow;

    await _context.SaveChangesAsync();
    return Ok(job);
}

public class UpdateJobRequest
{
    public string Status { get; set; }
    public string OutputKey { get; set; }
    public string ErrorMessage { get; set; }
}
```

### 3. Kafka Consumer

```csharp
public class KafkaConsumerService : BackgroundService
{
    private readonly IServiceProvider _serviceProvider;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers = "kafka:9092",
            GroupId = "backend-service",
            AutoOffsetReset = AutoOffsetReset.Earliest
        };

        using (var consumer = new ConsumerBuilder<string, string>(config).Build())
        {
            consumer.Subscribe(new[] { "job.completed", "job.failed" });

            while (!stoppingToken.IsCancellationRequested)
            {
                var result = consumer.Consume(stoppingToken);

                using (var scope = _serviceProvider.CreateScope())
                {
                    var db = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

                    if (result.Topic == "job.completed")
                    {
                        var data = JsonSerializer.Deserialize<JobCompletedMessage>(result.Message.Value);
                        var job = await db.Jobs.FindAsync(Guid.Parse(data.JobId));
                        if (job != null)
                        {
                            job.Status = "Completed";
                            job.OutputKey = data.OutputKey;
                            job.FinishedAt = DateTime.UtcNow;
                            await db.SaveChangesAsync();
                        }
                    }
                    else if (result.Topic == "job.failed")
                    {
                        var data = JsonSerializer.Deserialize<JobFailedMessage>(result.Message.Value);
                        var job = await db.Jobs.FindAsync(Guid.Parse(data.JobId));
                        if (job != null)
                        {
                            job.Status = "Failed";
                            job.ErrorMessage = data.Error;
                            job.FinishedAt = DateTime.UtcNow;
                            await db.SaveChangesAsync();
                        }
                    }
                }
            }
        }
    }
}

public class JobCompletedMessage { public string JobId { get; set; } public string OutputKey { get; set; } }
public class JobFailedMessage { public string JobId { get; set; } public string Error { get; set; } }
```

### 4. Download Endpoint

```csharp
[HttpGet("{id:guid}/download")]
public async Task<IActionResult> DownloadAsync(Guid id)
{
    var job = await _context.Jobs.FindAsync(id);
    if (job == null || string.IsNullOrEmpty(job.OutputKey))
        return NotFound();

    var stream = new MemoryStream();
    await minioClient.GetObjectAsync(new GetObjectArgs()
        .WithBucket("audio-files")
        .WithObject(job.OutputKey)
        .WithCallbackStream(async (s) => await s.CopyToAsync(stream)));

    stream.Seek(0);
    return File(stream, "audio/wav", $"output-{id}.wav");
}
```

### 5. Dependency Injection (Program.cs)

```csharp
services.AddSingleton<MinioClient>(_ => new MinioClient()
    .WithEndpoint("minio:9000")
    .WithCredentials("minio", "minio123")
    .Build());

services.AddSingleton<IProducer<string, string>>(_ =>
    new ProducerBuilder<string, string>(
        new ProducerConfig { BootstrapServers = "kafka:9092" })
    .Build());

services.AddHostedService<KafkaConsumerService>();
```

### 6. Message Format

Backend sends to Kafka topic `job.prepared`:

```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "inputKey": "input/job-550e8400-e29b-41d4-a716-446655440000.wav"
}
```

ML Service sends to `job.completed`:

```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "outputKey": "output/550e8400-e29b-41d4-a716-446655440000.wav"
}
```

Or `job.failed`:

```json
{
  "jobId": "550e8400-e29b-41d4-a716-446655440000",
  "error": "Failed to download from MinIO"
}
```

### Nuget Dependencies

```
Minio
Confluent.Kafka
```

public class JobFailedMessage
{
public string JobId { get; set; }
public string Error { get; set; }
}

````

---

## 3️⃣ PUT Endpoint для обновления Job

ML Service будет делать PUT запрос к этому endpoint'у:

```csharp
[HttpPut("{id:guid}")]
[ProducesResponseType(StatusCodes.Status200OK)]
[ProducesResponseType(StatusCodes.Status404NotFound)]
[ProducesResponseType(StatusCodes.Status400BadRequest)]
public async Task<IActionResult> UpdateJob(
    Guid id,
    [FromBody] UpdateJobRequest request
)
{
    var job = await _context.Jobs.FindAsync(id);

    if (job == null)
        return NotFound($"Job {id} not found");

    // Обновить поля
    if (!string.IsNullOrEmpty(request.Status))
        job.Status = request.Status;

    if (!string.IsNullOrEmpty(request.OutputKey))
        job.OutputKey = request.OutputKey;

    if (request.StartedAt.HasValue)
        job.StartedAt = request.StartedAt.Value;

    if (request.FinishedAt.HasValue)
        job.FinishedAt = request.FinishedAt.Value;

    job.UpdatedAt = DateTime.UtcNow;

    _context.Jobs.Update(job);
    await _context.SaveChangesAsync();

    return Ok(job);
}

public class UpdateJobRequest
{
    public string Status { get; set; }
    public string OutputKey { get; set; }
    public DateTime? StartedAt { get; set; }
    public DateTime? FinishedAt { get; set; }
}
````

---

## 4️⃣ GET Endpoint для скачивания результата

Когда пользователь захочет скачать обработанный файл:

```csharp
[HttpGet("{id:guid}/download")]
public async Task<IActionResult> DownloadResult(Guid id)
{
    var job = await _context.Jobs.FindAsync(id);

    if (job == null)
        return NotFound();

    if (string.IsNullOrEmpty(job.OutputKey))
        return BadRequest("Job not completed yet");

    // Скачать файл из MinIO
    var minioClient = new MinioClient()
        .WithEndpoint("minio:9000")
        .WithCredentials("minio", "minio123")
        .Build();

    try
    {
        var stream = new MemoryStream();

        var getArgs = new GetObjectArgs()
            .WithBucket("audio-files")
            .WithObject(job.OutputKey)
            .WithCallbackStream(async (s) => await s.CopyToAsync(stream));

        await minioClient.GetObjectAsync(getArgs);

        stream.Position = 0;

        return File(
            stream,
            "audio/wav",
            $"output-{job.Id}.wav"
        );
    }
    catch (Exception ex)
    {
        return StatusCode(500, ex.Message);
    }
}
```

---

## 5️⃣ Job модель в БД

```csharp
public class Job
{
    public Guid Id { get; set; }

    public string Status { get; set; } // Queued, Processing, Completed, Failed

    public string InputKey { get; set; }   // input/job-xxx.wav

    public string OutputKey { get; set; }  // output/job-xxx.wav

    public string ErrorMessage { get; set; }

    public DateTime CreatedAt { get; set; }

    public DateTime UpdatedAt { get; set; }

    public DateTime? StartedAt { get; set; }

    public DateTime? FinishedAt { get; set; }
}
```

---

## 6️⃣ Миграция Entity Framework

```csharp
modelBuilder.Entity<Job>(entity =>
{
    entity.HasKey(e => e.Id);

    entity.Property(e => e.Status)
        .IsRequired()
        .HasMaxLength(50);

    entity.Property(e => e.InputKey)
        .IsRequired()
        .HasMaxLength(500);

    entity.Property(e => e.OutputKey)
        .HasMaxLength(500);

    entity.Property(e => e.ErrorMessage)
        .HasMaxLength(1000);

    entity.HasIndex(e => e.Status);
    entity.HasIndex(e => e.CreatedAt);
});
```

---

## 7️⃣ Dependency Injection в Startup

```csharp
services.AddSingleton<IProducer<string, string>>(sp =>
{
    var config = new ProducerConfig { BootstrapServers = "kafka:9092" };
    return new ProducerBuilder<string, string>(config).Build();
});

services.AddSingleton<MinioClient>(sp =>
{
    return new MinioClient()
        .WithEndpoint("minio:9000")
        .WithCredentials("minio", "minio123")
        .Build();
});

// Запустить Kafka consumer в background
services.AddHostedService<KafkaConsumerService>();
```

---

## 8️⃣ Background Service для слушания Kafka

```csharp
public class KafkaConsumerService : BackgroundService
{
    private readonly ILogger<KafkaConsumerService> _logger;
    private readonly IServiceProvider _serviceProvider;

    public KafkaConsumerService(
        ILogger<KafkaConsumerService> logger,
        IServiceProvider serviceProvider
    )
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers = "kafka:9092",
            GroupId = "backend-service",
            AutoOffsetReset = AutoOffsetReset.Earliest,
        };

        using (var consumer = new ConsumerBuilder<string, string>(config).Build())
        {
            consumer.Subscribe(new[] { "job.completed", "job.failed" });

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    var consumeResult = consumer.Consume(stoppingToken);

                    using (var scope = _serviceProvider.CreateScope())
                    {
                        var dbContext = scope.ServiceProvider.GetRequiredService<ApplicationDbContext>();

                        if (consumeResult.Topic == "job.completed")
                        {
                            var data = JsonSerializer.Deserialize<JobCompletedMessage>(
                                consumeResult.Message.Value
                            );

                            var job = await dbContext.Jobs.FindAsync(Guid.Parse(data.JobId));
                            if (job != null)
                            {
                                job.Status = "Completed";
                                job.OutputKey = data.OutputKey;
                                job.FinishedAt = DateTime.UtcNow;
                                await dbContext.SaveChangesAsync();

                                _logger.LogInformation($"Job {data.JobId} completed");
                            }
                        }
                        else if (consumeResult.Topic == "job.failed")
                        {
                            var data = JsonSerializer.Deserialize<JobFailedMessage>(
                                consumeResult.Message.Value
                            );

                            var job = await dbContext.Jobs.FindAsync(Guid.Parse(data.JobId));
                            if (job != null)
                            {
                                job.Status = "Failed";
                                job.ErrorMessage = data.Error;
                                job.FinishedAt = DateTime.UtcNow;
                                await dbContext.SaveChangesAsync();

                                _logger.LogError($"Job {data.JobId} failed: {data.Error}");
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error processing Kafka message");
                }
            }
        }
    }
}
```

---

## ✅ Checklist для интеграции

- [ ] Установлены NuGet пакеты: `Minio`, `Confluent.Kafka`
- [ ] Добавлена Job модель в БД
- [ ] Создана миграция для таблицы Jobs
- [ ] Реализован POST endpoint для загрузки файла
- [ ] Реализован PUT endpoint `/api/jobs/{id}` для обновления статуса
- [ ] Реализован GET endpoint для скачивания результата
- [ ] Настроен Kafka Consumer в BackgroundService
- [ ] Конфигурация Kafka и MinIO в appsettings.json
- [ ] Протестирована整합 с ML Service

---

**Готово! Backend теперь полностью интегрирован с ML Service архитектурой.**
