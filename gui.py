import streamlit as st
import numpy as np
from perceptron import train_perceptron, predict, forward_prop
from multi_perceptron import train_ann, predict_ann


def main():
    st.title("Perceptron/ANN Logic Gate Trainer & Tester")
    st.write(
        "Train a perceptron or a 2-layer ANN (for XOR/XNOR) on the selected logic gate and test it with custom inputs."
    )

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    gate_type = st.sidebar.radio(
        "Select Logic Gate",
        ["AND", "NAND", "OR", "NOR", "XOR", "XNOR", "NOT"],
        help="Choose which logic gate to train the perceptron/ANN on",
        index=0,
    )
    gate_type_internal = gate_type.lower()

    learning_rate = st.sidebar.slider(
        "Learning Rate",
        min_value=0.01,
        max_value=0.5,
        value=0.1,
        step=0.01,
        help="Learning rate for gradient descent",
    )

    num_iterations = st.sidebar.slider(
        "Number of Training Iterations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        help="Number of training epochs",
    )

    # Training section
    st.header("Training")

    if st.button("Train Model"):
        with st.spinner("Training model..."):
            if gate_type_internal in ["xor", "xnor"]:
                result = train_ann(gate_type_internal, learning_rate, num_iterations)
                st.session_state.model_type = "ann"
                st.session_state.W1 = result["W1"]
                st.session_state.b1 = result["b1"]
                st.session_state.W2 = result["W2"]
                st.session_state.b2 = result["b2"]
                st.session_state.gate_type = gate_type
                st.session_state.trained = True
                st.session_state.df = result["df"]
                st.session_state.preds = result["preds"]
                st.session_state.Y = result["Y"]
                st.session_state.X = result["X"]
                st.success(f"ANN trained successfully on {gate_type} gate!")
                st.info("A 2-layer ANN with 2 hidden neurons and ReLU activations was used.")
            else:
                result = train_perceptron(gate_type_internal, learning_rate, num_iterations)
                st.session_state.model_type = "perceptron"
                st.session_state.W = result["W"]
                st.session_state.b = result["b"]
                st.session_state.gate_type = gate_type
                st.session_state.trained = True
                st.session_state.df = result["df"]
                st.session_state.preds = result["preds"]
                st.session_state.Y = result["Y"]
                st.session_state.X = result["X"]
                st.success(f"Perceptron trained successfully on {gate_type} gate!")
            # Display training results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{result['accuracy']:.2%}")
            with col2:
                st.metric("Final Loss", f"{result['loss']:.4f}")
            # Display truth table
            st.subheader("Training Results (Truth Table)")
            results_df = result["df"].copy()
            results_df["Predicted"] = result["preds"]
            results_df["Correct"] = result["preds"] == result["Y"]
            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

    # Testing section
    st.header("Testing")

    if "trained" in st.session_state and st.session_state.trained:
        st.write(
            f"Model trained on {st.session_state.gate_type} gate is ready for testing!"
        )
        # Input section
        st.subheader("Enter Test Inputs")
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.selectbox("Input 1 (x1)", [0, 1], help="First binary input")
        with col2:
            x2 = st.selectbox("Input 2 (x2)", [0, 1], help="Second binary input")
        if st.button("Test Model"):
            X_test = np.array([[x1, x2]])
            if st.session_state.model_type == "ann":
                prediction = predict_ann(
                    X_test,
                    st.session_state.W1,
                    st.session_state.b1,
                    st.session_state.W2,
                    st.session_state.b2,
                )[0][0]
                # For ANN, also show raw output
                from multi_perceptron import forward_prop
                _, _, raw_output, _ = forward_prop(
                    X_test,
                    st.session_state.W1,
                    st.session_state.b1,
                    st.session_state.W2,
                    st.session_state.b2,
                )
                raw_output = raw_output[0][0]
            else:
                prediction = predict(X_test, st.session_state.W, st.session_state.b)[0]
                raw_output, _ = forward_prop(X_test, st.session_state.W, st.session_state.b)
                raw_output = raw_output[0]
            st.subheader("Test Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input 1", x1)
            with col2:
                st.metric("Input 2", x2)
            with col3:
                st.metric("Prediction", prediction)
            st.write(f"**Raw Output (before thresholding):** {raw_output:.4f}")
            # Expected output based on gate type
            if st.session_state.gate_type == "AND":
                expected = 1 if x1 == 1 and x2 == 1 else 0
            elif st.session_state.gate_type == "NAND":
                expected = 0 if x1 == 1 and x2 == 1 else 1
            elif st.session_state.gate_type == "OR":
                expected = 1 if x1 == 1 or x2 == 1 else 0
            elif st.session_state.gate_type == "NOR":
                expected = 1 if x1 == 0 and x2 == 0 else 0
            elif st.session_state.gate_type == "XOR":
                expected = 1 if x1 != x2 else 0
            elif st.session_state.gate_type == "XNOR":
                expected = 1 if x1 == x2 else 0
            else:  # NOT
                expected = 1 - x1
            st.write(f"**Expected Output:** {expected}")
            st.write(f"**Correct:** {'✅' if prediction == expected else '❌'}")
            st.subheader("Visual Representation")
            if prediction == 1:
                st.success("Output: 1 (True)")
            else:
                st.info("Output: 0 (False)")
    else:
        st.info(
            "Please train the model first using the 'Train Model' button above."
        )
    # Information section
    with st.expander("About Logic Gates"):
        st.write(
            """
        **AND Gate:** Outputs 1 only when both inputs are 1, otherwise outputs 0.

        **NAND Gate:** Outputs 0 only when both inputs are 1, otherwise outputs 1 (NOT AND).

        **OR Gate:** Outputs 1 if at least one input is 1, otherwise outputs 0.

        **NOR Gate:** Outputs 1 only when both inputs are 0, otherwise outputs 0 (NOT OR).

        **XOR Gate:** Outputs 1 if the inputs are different, otherwise outputs 0 (exclusive OR).

        **XNOR Gate:** Outputs 1 if the inputs are the same, otherwise outputs 0 (exclusive NOR).

        **NOT Gate:** Outputs 1 if the input is 0, and 0 if the input is 1 (inverts a single input).

        The perceptron or ANN learns these patterns through training on the truth table data.
        """
        )


if __name__ == "__main__":
    main()
