import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const runPythonInference = (testDataPath, testLabelPath) => {
    return new Promise((resolve, reject) => {
        const pythonExecutablePath = path.join(
            __dirname,
            "myenv",
            "bin",
            "python3.11"
        );

        const pythonProcess = spawn(pythonExecutablePath, [
            "./model/main_testing.py",
            testDataPath,
            testLabelPath,
            "./model/model_weights.pth",
        ]);

        pythonProcess.stdout.on("data", (data) => {
            console.log(`stdout: ${data}`);
            resolve(data.toString());
        });

        pythonProcess.stderr.on("data", (data) => {
            console.error(`stderr: ${data}`);
            reject(data.toString());
        });
    });
};

export default runPythonInference;
