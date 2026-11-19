# AI Use Disclosure (DSA 8401)

## 1. Tool and Scope of AI Use

**AI tool:** Claude Code (Anthropic), accessed through the Claude desktop application using the Claude Sonnet 5.5 model.

Claude Code was used as a supporting tool during the development of this capstone project, primarily for software implementation, debugging, testing, documentation, and presentation preparation.

The project was student-directed, with the research objectives, project requirements, methodological decisions, evaluation oversight, and final deliverables managed by me.

## 2. My Personal Contributions

### Project conceptualization and research direction

I independently identified and selected the research topic of customer inactivity prediction, drawing on my MSc thesis work and interests in predictive analytics.

I defined the central research problem and established customer inactivity prediction as the primary objective, while identifying the combined assessment of inactivity and credit risk as a potential future research direction.

I also decided to use publicly available data rather than confidential organizational information.

### Dataset selection and research methodology

I selected the UCI Online Retail II dataset from the available alternatives based on its relevance to customer transaction behaviour and inactivity prediction.

I directed the methodological requirements and reviewed the validation design, including chronological data splitting and the use of a label-maturity embargo to reduce potential data leakage.

I established the modelling objectives, evaluation expectations, and project deliverables.

### Project development and requirements management

I directed the development of the project's core functionality and identified additional capabilities required to improve its analytical scope.

These included:

* Data drift monitoring and alerting.
* Support Vector Machine (SVM) modelling.
* K-Means customer segmentation.
* GitHub Actions continuous integration.
* Statistical hypothesis testing.
* Comparative feature-selection analysis.
* Isolation Forest anomaly detection.

I assessed the project's evolving requirements and directed subsequent enhancements to ensure alignment with the research objectives.

### Evaluation and quality assurance

I reviewed the generated evaluation reports, model performance metrics, statistical findings, and supporting visualizations.

Particular attention was given to the model's predictive performance, comparison with baseline approaches, calibration behaviour, and limitations arising from changes in the underlying data distribution.

I reviewed the testing evidence and project outputs to assess their consistency with the intended research requirements.

### Repository management and final submission

I managed the project's publication, configured the GitHub repository for public access, and verified that the GitHub Actions workflow runs passed following publication.

I maintained responsibility for the project's overall direction, acceptance of deliverables, and final submission.

## 3. How AI Was Used

Claude Code provided technical assistance throughout the project.

### Software implementation

AI generated and refined the Python source code within `src/inactivity/`, including components responsible for data processing, feature engineering, predictive modelling, evaluation, explainability, drift monitoring, customer segmentation, statistical testing, feature selection, and anomaly detection.

It also generated the automated tests, Docker configuration files, the GitHub Actions workflow, and the API-served web interface.

### Documentation and reporting

AI drafted and refined the capstone proposal, `README.md`, technical documentation in `docs/`, the final report, presentation slides, and supporting figures.

### Debugging and testing

AI was used to assist with troubleshooting implementation issues, running pipeline components, executing tests, and identifying possible improvements.

AI also suggested elements of the validation methodology, which were reviewed in relation to the project's requirements.

Generated outputs were incorporated into the project following review and acceptance.

## 4. Verification and Validation

The project included 63 automated tests covering data cleaning, leakage prevention, chronological splitting, embargo enforcement, evaluation metric consistency, drift detection, statistical testing, API functionality, and training-serving consistency.

Testing was conducted using `python -m pytest`.

The pipeline was initially executed using synthetic data, followed by the UCI Online Retail II dataset in October 2026.

Additional statistical analyses, feature-selection comparisons, and anomaly checks were conducted using the real dataset in November 2026.

The resulting outputs were retained in `reports/` for examination and reference.

I reviewed the model evaluation findings, including results that demonstrated limited improvements over simpler baseline models and calibration challenges caused by distribution changes.

## 5. Limitations and Quality Assurance

Automated tests provide evidence that the implemented code behaves according to its specified requirements. However, successful tests do not independently establish that the modelling choices are optimal.

The Docker image was built and run on one machine and in GitHub Actions, which limits conclusions about deployment portability to other environments. The GitHub Actions workflow runs passed after the repository was published.

The project also identified limitations in predictive performance and probability calibration. These findings were retained in the documentation to ensure balanced reporting.

## 6. Data Protection and Privacy

The project uses the publicly available UCI Online Retail II dataset, distributed under the CC BY 4.0 licence.

No confidential organizational information or proprietary company datasets were incorporated into the project.

The project was developed using publicly available research data, with appropriate consideration given to customer identifier fields and data handling.

## 7. Author's Declaration

I declare that this capstone project was undertaken under my direction and supervision, with AI used as a supporting tool during selected technical and documentation activities.

I independently established the research topic, defined the problem and objectives, selected the dataset, directed project requirements and enhancements, reviewed the methodology and results, and managed the final submission.

AI assistance contributed to software implementation, automated testing, technical documentation, debugging, and presentation development, as disclosed above.

I acknowledge that AI-generated material requires appropriate review and that responsibility for the submitted work remains with me.

I take responsibility for the project's research direction, methodological decisions, evaluation findings, documented limitations, and final deliverables.

I understand that this project is individually assessed and that I am expected to explain and defend its methodology, implementation, results, and conclusions.

**Student signature:** <img src="docs/figures/signature.png" alt="Signature of Eric Muriuki Kiama" width="130">

**Name:** Eric Muriuki Kiama (120890)

**Date:** 19 November 2026
