package model

type RouteRequest struct {
	Places []string `json:"places"` // Список категорий
	Time   string   `json:"time"`   // Время прогулки
	Budget string   `json:"budget"` // Бюджет
}

type Place struct {
	Name        string    `json:"name"`        // Название места
	Coordinates []float64 `json:"coordinates"` // Координаты
}
