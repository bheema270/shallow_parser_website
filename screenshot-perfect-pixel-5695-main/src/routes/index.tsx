import { createFileRoute, redirect, Navigate } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  beforeLoad: () => {
    throw redirect({
      to: "/shallow_parser",
    });
  },
  component: () => <Navigate to="/shallow_parser" />,
});
