"use client";

import { useState } from "react";
import { Exercise } from "@/formcheck/types";
import FormCheckCamera from "@/formcheck/components/FormCheckCamera";

export default function FormCheckPage() {
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);

  return (
    <div className="text-foreground font-sans selection:bg-primary/30 selection:text-primary p-2 sm:p-3 md:p-4 lg:p-4">
      <FormCheckCamera
        exercise={selectedExercise}
        onSelectExercise={setSelectedExercise}
      />
    </div>
  );
}
