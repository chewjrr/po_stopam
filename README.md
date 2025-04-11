# Po Stopam Backend

Backend-часть приложения для генерации маршрутов прогулок по Москве. Предоставляет API для получения списка мест с координатами.

## 📋 Требования
- Go 1.21+
- Python 3.8+ (для работы нейромодели)
- Git

## 🚀 Быстрый старт

### Установка
```bash
git clone -b backend https://github.com/chewjrr/po_stopam.git
cd po_stopam
go mod download
```

### Запуск
```bash
go run ./cmd/main.go
```

Сервер будет доступен на `http://localhost:8080`

### Тестовый запрос
```bash
curl -X POST http://localhost:8080/generate-route \
  -H "Content-Type: application/json" \
  -d '{
    "places": ["парк", "кафе"],
    "time": "2 часа",
    "budget": "1500 рублей"
  }'
```

## 🌐 API Endpoints

### 1. Полный формат маршрута
**POST** `/generate-route`
```json
{
  "places": ["string"],
  "time": "string",
  "budget": "string"
}
```
Пример ответа:
```json
[
  {
    "name": "Парк Горького",
    "coordinates": [55.7270939, 37.6002408]
  }
]
```

### 2. Только координаты
**POST** `/generate-coordinates`  
(тело запроса аналогично)

Пример ответа:
```json
[
  [55.7270939, 37.6002408],
  [55.7044118, 37.5340531]
]
```

## 📂 Структура проекта
```
.
├── cmd/              # Точка входа
├── internal/
│   ├── handler/      # HTTP обработчики
│   ├── model/        # Структуры данных
│   └── service/      # Бизнес-логика
├── scripts/          # Python-скрипты
└── go.mod            # Зависимости Go
```

## 📄 Документация
Swagger UI доступен после запуска сервера:  
http://localhost:8080/swagger/index.html
