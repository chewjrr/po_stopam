package service

import (
	"AslanBackEndUgli/internal/model"
	"encoding/json"
	"fmt"
	"os/exec"
)

func GenerateRoute(req model.RouteRequest) ([]model.Place, error) {
	// Сериализуем запрос в JSON для передачи в Python
	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("json encode error: %v", err)
	}

	// Вызываем Python-скрипт
	cmd := exec.Command("python3", "./scripts/model.py", string(jsonData))

	// Получаем вывод скрипта
	output, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("python error: %v\nOutput: %s", err, string(output))
	}

	// Парсим результат
	var route []model.Place
	if err := json.Unmarshal(output, &route); err != nil {
		return nil, fmt.Errorf("json decode error: %v", err)
	}

	return route, nil
}
