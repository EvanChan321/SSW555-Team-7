import { spawn } from "child_process";

const runPythonInference = (eegData) => {
    return new Promise((resolve, reject) => {
        const dataString = JSON.stringify(eegData);

        const pythonProcess = spawn("python", [
            "main.py",
            "--data",
            dataString,
        ]);

        pythonProcess.stdout.on("data", (data) => {
            resolve(data.toString());
        });

        pythonProcess.stderr.on("data", (data) => {
            reject(data.toString());
        });
    });
};

export default runPythonInference;
