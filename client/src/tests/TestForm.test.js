import React from "react";
import { render, fireEvent, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import TestForm from "../components/TestForm";
import FileUploadForm from "../components/FileUploadForm";
import ResultsDisplay from "../components/ResultsDisplay";

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

    test("renders file upload inputs and button", () => {
        render(<FileUploadForm />);
        expect(screen.getByLabelText("Upload Test Data:")).toBeInTheDocument();
        expect(screen.getByLabelText("Upload Test Labels:")).toBeInTheDocument();
        expect(screen.getByRole("button", { name: "Process Data" })).toBeInTheDocument();
    });

    test("handles changes in file inputs", () => {
        render(<FileUploadForm />);
        const testDataInput = screen.getByLabelText("Upload Test Data:");
        const testLabelInput = screen.getByLabelText("Upload Test Labels:");

        fireEvent.change(testDataInput, {
            target: { files: [new File(["test data"], "testData.txt")] },
        });
        fireEvent.change(testLabelInput, {
            target: { files: [new File(["test labels"], "testLabels.txt")] },
        });

        expect(testDataInput.files[0]).toBeDefined();
        expect(testLabelInput.files[0]).toBeDefined();
    });

    test("renders results when provided", () => {
        const results = "Test Result";
        render(<ResultsDisplay results={results} />);
        expect(screen.getByText("Results:")).toBeInTheDocument();
        expect(screen.getByText(results)).toBeInTheDocument();
    });
});
