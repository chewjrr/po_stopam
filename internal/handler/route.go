package handler

import (
	"AslanBackEndUgli/internal/model"
	"AslanBackEndUgli/internal/service"
	"encoding/json"
	"net/http"
)

// @Summary Генерирует маршрут с местами и координатами
// @Description Принимает параметры для генерации маршрута и возвращает список мест с координатами
// @Tags Маршруты
// @Accept json
// @Produce json
// @Param request body model.RouteRequest true "Параметры запроса"
// @Success 200 {array} model.Place
// @Failure 400 {string} string "Неверный формат запроса"
// @Failure 500 {string} string "Ошибка генерации маршрута"
// @Router /generate-route [post]
func RouteHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req model.RouteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}

	route, err := service.GenerateRoute(req)
	if err != nil {
		http.Error(w, "Error generating route: "+err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(route)
}

// @Summary Генерирует только координаты маршрута
// @Description Принимает параметры для генерации маршрута и возвращает только координаты
// @Tags Маршруты
// @Accept json
// @Produce json
// @Param request body model.RouteRequest true "Параметры запроса"
// @Success 200 {array} array{float64} "Пример: [[55.7270939,37.6002408], ...]"
// @Failure 400 {string} string "Неверный формат запроса"
// @Failure 500 {string} string "Ошибка генерации маршрута"
// @Router /generate-coordinates [post]
func CoordinatesHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req model.RouteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}

	route, err := service.GenerateRoute(req)
	if err != nil {
		http.Error(w, "Error generating route: "+err.Error(), http.StatusInternalServerError)
		return
	}

	coords := make([][]float64, 0)
	for _, place := range route {
		coords = append(coords, place.Coordinates)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(coords)
}
