import { useEffect, useState } from "react";
import { API_BASE } from "@/config";
import type { ModelOption } from "@/types/parser";

export function useModels() {
  const [models, setModels] = useState<ModelOption[]>([
    { id: "v1", label: "MuRIL MTL V1 (Curated 32K)", is_default: true, model_file: "best_model_v1.pt" },
    { id: "v8", label: "MuRIL MTL V8 (Raw Baseline 213K)", is_default: false, model_file: "best_model_v8.pt" },
    { id: "v9", label: "MuRIL MTL V9 (Deduplicated 121K)", is_default: false, model_file: "best_model_v9.pt" },
  ]);
  const [selectedModel, setSelectedModel] = useState<string>("v1");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;
    const fetchModels = async () => {
      setLoading(true);
      try {
        const res = await fetch(`${API_BASE}/api/models`);
        if (res.ok) {
          const data = await res.json();
          if (active && Array.isArray(data.models) && data.models.length > 0) {
            setModels(data.models);
            const defaultId = data.default_model || data.models[0].id;
            setSelectedModel((prev) => (data.models.some((m: ModelOption) => m.id === prev) ? prev : defaultId));
          }
        }
      } catch (err) {
        console.warn("Could not fetch models from backend, using fallback default", err);
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchModels();
    return () => {
      active = false;
    };
  }, []);

  return { models, selectedModel, setSelectedModel, loading };
}
