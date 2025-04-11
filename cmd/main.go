package main

import (
	_ "AslanBackEndUgli/docs" // Импорт сгенерированной документации
	"AslanBackEndUgli/internal/handler"
	"github.com/swaggo/http-swagger" // Добавьте этот импорт
	"log"
	"net/http"
)

func main() {
	// Эндпоинты API
	http.HandleFunc("/generate-route", handler.RouteHandler)
	http.HandleFunc("/generate-coordinates", handler.CoordinatesHandler)

	// Документация Swagger
	http.Handle("/swagger/", httpSwagger.WrapHandler)

	log.Println("Server started on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
