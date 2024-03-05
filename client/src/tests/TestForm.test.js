import React from "react";
import { render, fireEvent, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TestForm from "../components/TestForm";

describe("TestForm", () => {
    test("renders input and button", () => {
        render(<TestForm onSubmit={() => {}} />);
        expect(screen.getByLabelText(/Enter Text:/i)).toBeInTheDocument();
        expect(
            screen.getByRole("button", { name: /submit/i })
        ).toBeInTheDocument();
    });

    test("calls onSubmit with the input value when form is submitted", () => {
        const mockSubmit = jest.fn();
        render(<TestForm onSubmit={mockSubmit} />);

        fireEvent.change(screen.getByLabelText(/Enter Text:/i), {
            target: { value: "test value" },
        });
        fireEvent.click(screen.getByRole("button", { name: /submit/i }));

        expect(mockSubmit).toHaveBeenCalledWith("test value");
    });
});
