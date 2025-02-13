import  { useRef, useState } from "react";
import { ReactSketchCanvas } from "react-sketch-canvas";

export default function DigitCanvas() {
    const [prediction, setPrediction] = useState("Prediction: 3");
    const canvasRef = useRef(null);

    const clearCanvas = () => {
        canvasRef.current.clearCanvas(); // Clear canvas
    };

    const classify = async () => {
        const imageData = await canvasRef.current.exportImage("png"); // Export image data
console.log(imageData);
        const response = await fetch("#", {
            method: "POST",
            body: JSON.stringify({ image: imageData }),
            headers: { "Content-Type": "application/json" }
        });

        const result = await response.json();
        setPrediction("Prediction: " + result.prediction);
    };

    return (
        <div className="bg-gray-900  min-h-screen flex flex-col items-center justify-center text-white font-bold">
            <div className="relative md:w-150 w-90 aspect-square flex flex-col items-center justify-center p-1 py-4 rounded-lg shadow-xl bg-gray-800 border-4 border-gray-700">
                <h2 className="text-3xl p-1 md:text-4xl font-bold text-center text-gradient mb-4 text-transparent bg-clip-text bg-white">
                    Draw a Digit
                </h2>
                <ReactSketchCanvas
                    ref={canvasRef}
                    width="500"
                    height="500"
                    strokeColor="black"
                    strokeWidth={6}
                    className="border-4 w-[90%]  h-[80%] border-gray-700 rounded-lg shadow-xl"
                />
                <div className="mt-6 space-x-4">
                    <button
                        onClick={clearCanvas}
                        className="bg-red-600 hover:bg-red-700 text-white py-1.5 px-7 rounded-lg text-lg font-semibold transition duration-300 shadow-lg"
                    >
                        Clear
                    </button>
                    <button
                        onClick={classify}
                        className="bg-green-600 hover:bg-green-700 text-white py-1.5 px-7 rounded-lg text-lg font-semibold transition duration-300 shadow-lg"
                    >
                        Detect
                    </button>
                </div>
                <p className="mt-6 text-xl font-bold drop-shadow-lg">{prediction}</p>
            </div>
        </div>
    );
}
