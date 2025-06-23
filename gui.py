import streamlit as st
import numpy as np
from perceptron import _init_data, _init_params, forward_prop, train, predict


def main():
    st.title("Perceptron Logic Gate Trainer & Tester")
    st.write(
        "Train a perceptron on AND or NAND logic gates and test it with custom inputs."
    )

    # Sidebar for configuration
    st.sidebar.header("Configuration")
    gate_type = st.sidebar.selectbox(
        "Select Logic Gate",
        ["AND", "NAND"],
        help="Choose which logic gate to train the perceptron on",
    )

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

    if st.button("Train Perceptron"):
        with st.spinner("Training perceptron..."):
            # Initialize data and parameters
            df = _init_data(gate_type.lower())
            X = df[["x1", "x2"]].values
            Y = df["y"].values
            W, b = _init_params(X.shape[1])

            # Train the model
            W_trained, b_trained = train(X, Y, W, b, learning_rate, num_iterations)

            # Store trained parameters in session state
            st.session_state.W = W_trained
            st.session_state.b = b_trained
            st.session_state.gate_type = gate_type
            st.session_state.trained = True

            st.success(f"Perceptron trained successfully on {gate_type} gate!")

            # Display training results
            preds = predict(X, W_trained, b_trained)
            accuracy = np.mean(preds == Y)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{accuracy:.2%}")
            with col2:
                st.metric(
                    "Final Loss",
                    f"{np.mean((forward_prop(X, W_trained, b_trained)[0] - Y) ** 2):.4f}",
                )

            # Display truth table
            st.subheader("Training Results (Truth Table)")
            results_df = df.copy()
            results_df["Predicted"] = preds
            results_df["Correct"] = preds == Y
            st.dataframe(results_df, use_container_width=True)

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

        if st.button("Test Perceptron"):
            # Prepare test input
            X_test = np.array([[x1, x2]])

            # Get prediction
            prediction = predict(X_test, st.session_state.W, st.session_state.b)[0]

            # Get raw output (before thresholding)
            raw_output, _ = forward_prop(X_test, st.session_state.W, st.session_state.b)

            # Display results
            st.subheader("Test Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Input 1", x1)
            with col2:
                st.metric("Input 2", x2)
            with col3:
                st.metric("Prediction", prediction)

            # Show raw output
            st.write(f"**Raw Output (before thresholding):** {raw_output[0]:.4f}")

            # Expected output based on gate type
            if st.session_state.gate_type == "AND":
                expected = 1 if x1 == 1 and x2 == 1 else 0
            else:  # NAND
                expected = 0 if x1 == 1 and x2 == 1 else 1

            st.write(f"**Expected Output:** {expected}")
            st.write(f"**Correct:** {'✅' if prediction == expected else '❌'}")

            # Visual representation
            st.subheader("Visual Representation")
            if prediction == 1:
                st.success("Output: 1 (True)")
            else:
                st.info("Output: 0 (False)")

    else:
        st.info(
            "Please train the perceptron first using the 'Train Perceptron' button above."
        )

    # Information section
    with st.expander("About Logic Gates"):
        st.write("""
        **AND Gate:** Outputs 1 only when both inputs are 1, otherwise outputs 0.
        
        **NAND Gate:** Outputs 0 only when both inputs are 1, otherwise outputs 1 (NOT AND).
        
        The perceptron learns these patterns through training on the truth table data.
        """)


if __name__ == "__main__":
    main()
