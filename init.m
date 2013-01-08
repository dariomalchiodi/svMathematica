(* ::Package:: *)

(******************************************************************************
       Copyright (C) 2010 Dario Malchiodi <malchiodi@di.unimi.it>

This file is part of svMathematica.
svMathematica is free software; you can redistribute it and/or modify it under the
terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 2.1 of the License, or (at your option)
any later version.
svMathematica is distributed in the hope that it will be useful, but without any
warranty; without even the implied warranty of merchantability or fitness
for a particular purpose. See the GNU Lesser General Public License for
more details.
You should have received a copy of the GNU Lesser General Public License
along with svMathematica; if not, see <http://www.gnu.org/licenses/>.

******************************************************************************)

(* : Title : svMathematica *)

(* : Context : svMathematica *)

(* : Author : Dario Malchiodi *)

(* : Summary : This is an implementation of SVM *)

(* : Package Version : 0.8 *)

(* : Mathematica Version : 6.0 *)

(* : Keywords : SVM, Machine learning *)

(* : ToDo: parameter selection, LOO, cross-validation, error bounds, kernel matrix values, clustering, data set formats
*)









BeginPackage["svMathematica`"]

(* Unprotect the public symbols for functions and options *)
Unprotect[
  c,
  kernel,
  bVarThreshold,
  implementation,
  verbose,
  positiveColor,
  negativeColor,
  positiveSize,
  negativeSize,
  svmClassification,
  svmClassificationMaximize,
  svmClassificationAMPL,
  svmClassificationPython,
  svmClassificationSVMLight, 	
  svmGetClassifier,
  svmClassificationSamplePlot,
  svmRegressionSamplePlot,
  svmRegression,
  svmRegressionMaximize,
  svmRegressionAMPL,
  svmRegressionPython,
  lambda,
  matrixInverter,
  pythonInverter,
  classifierOutput,
  classifierInput,
  real,
  pattern
];

svmClassification::usage = "svmClassification[x, y] returns a classifier for points in x whose labels are in y";
svmClassificationMaximize::usage = "svmClassificationMaximize[x, y] returns a classifier for points in x whose labels are in y; optimization problem is handled through NMaximize ";
svmClassificationAMPL::usage = "svmClassificationAmpl[x, y] returns a classifier for patterns in x whose labels are in y; optimizazion problem is handled through AMPL/snopt";
svmClassificationPython::usage = "svmClassificationPyton[x, y] returns a classifier for patterns in x whose labels are in y; optimizazion problem is handled through Python/cvxopt";
svmClassificationSVMLight::usage = "svmClassificationSVMLight[x, y] returns a classifier for patterns in x whose labels are in y; optimizazion problem is handled through SVMLight";
svmGetClassifier::usage = "svmGetClassifier[x, y, a] returns a classifier for patterns in x whose labels are in y, using values in a as optimal lagrange multiplier values";
svmClassificationSamplePlot::usage = "svmClassificationSamplePlot[x, y] returns a graphic object corresponding to the scatter plot of patterns in x colored according to label values in y";
svmRegressionSamplePlot::"usage" = "svmRegressionSamplePlot[x, y] returns a graphic object corresponding to the scatter plot of patterns in x, augmented with labels in y";
svmDecisionFunctionPlot::usage = "svmDecisionFunctionPlot[f, {x, xmin, xmax} [,{y, ymin, ymax} [,{z, zmin, zmax}]]] returns a graphic object corresopnding to the decision function plot f";
svmRegression::usage = "svmRegression[x, y] returns a regressor for patterns in x whose labels are in y";
svmRidgeRegression::usage = "svmRidgeRegression[x, y] returns a ridge regressor for patterns in x whose labels are in y";
svmLinearInsensitiveRegression::usage = "svmLinearInsensitiveRegression[x, y] returns a linear epsilon-insensitive regressor for patterns in x whose labels are in y";

Options[svmClassification]={
  c->Infinity,                             (* default value for the parameter C of SVM classification *)
  kernel->"linear",                        (* default kernel description *)
  bVarThreshold->0.0001,                   (* threshold on variance of obtained intercept values *)
  implementation->svmClassificationPython, (* default implementation of the optimization solver procedure *)
  verbose->False,                          (* default verbosity *)
  classifierOutput->"real",                 (* output of the procedure *)
  classifierInput->"pattern"};              (* input of the decision function *)





Options[svmRegression]={
  kernel->"linear",                        (* default kernel description *)
  implementation->svmRidgeRegression,      (* default implementation of svmRegression *)
  verbose->False};                         (* default verbosity *)

Options[svmRidgeRegression]={
  lambda->1,                                (* default value for the parameter lambda of SVM ridge regression *)
  matrixInverter->Inverse};                 (* default implementation of the matrix inversion procedure *)

Options[svmLinearInsensitiveRegression]={
  epsilon->.1,                              (* default value for the parameter epsilon of SVM insensitive regression *)
  c->100,                                   (* default value for the parameter C of SVM insensitive regression *)
  optimizer->svmPythonOptimizer};           (* default implementation of the optimization solver procedure *)

Options[svmClassificationSamplePlot]={
  positiveColor->Green,                    (* default color for positive points *)
  negativeColor->Blue,                     (* default color for negative points *)
  positiveSize->5,                          (* default size for positive points *)
  negativeSize->5};                         (* default size for negative points *)

Options[svmDecisionFunctionPlot]={
  frontier->True,                               (* default visualization of frontier *)
  margin->False,                                (* default visualization of margins *)
  shading->False,                               (* default visualization of shading *)
  frontierStyle->{Black,AbsoluteThickness[1]},  (* default frontier style *)
  marginStyle->{Black},                         (* default margin style *)
  frontier3DStyle->{Blue,Opacity[.5]},          (* default 3D frontier style *)
  margin3DStyle->{Gray,Opacity[.1]},            (* default 3D margin style *)
  shadingColors->{positiveColor,negativeColor}/.Options[svmClassificationSamplePlot],  (* default shading colors *)
  shadingContours->100                          (* default number of shading components*)
};











(* ::Section:: *)
(*Package inizialization*)


Begin["`Private`"]

svm::unequalLength="patterns (`1`) and labels (`2`) have different length";
svm::unequalLengthAlpha="patterns (`1`), labels (`2`) and multipliers (`3`) do not have a same length";
svm::unknownKernel="unknown kernel specification `1`";
svm::exceedBVarThreshold="b variance exceeds threshold `1` (computed values `2`)";
svm::wrongLabelSymbols="labels list should contain only 1s and -1s";
svm::unbalancedLabels="labels list shoult contain at least one 1 and one -1";
svm::invalidPatternDimension="only patterns of dimension 1, 2, or 3 can be visualized";
svm::AMPLUnavailable="AMPL/snopt unavailable";
svm::PythonUnavailable="Python/cvxopt unavailable";
svm::SVMLightUnavailable="SVMLight unavailable";
svm::invalidOptionValue = "invalid value `2`for option `1`";

(* Check availability of AMPL/SNOPT and Python/cvxopt *)
svmAMPLAvailable = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> ":$PATH;ampl -v?" ] == 0;
svmPythonAvailable = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> 
    ":$PATH;python " <> $UserBaseDirectory <> "/Applications/svMathematica/" <> "test.py" ] == 0;
svmSVMLightAvailable = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> 
    ":$PATH;svm_learn " <> $UserBaseDirectory <> "/Applications/svMathematica/" <> "svm3000.dat" ] == 0;











(* ::Section:: *)
(*Utilities*)


(* svmClassificationLabelsQ checks wether or not the specified argument
   is a list of classification labels, that is only 1s and -1s occur
   in the list itself.

   returns: True if the specified argument is a list of classification labels,
            False otherwise.
*)
svmClassificationLabelsQ[
    labels_  (* labels list to be chekced *)
  ]:=Block[
  {cp, (* number of 1s *)
   cm  (* number of -1s *) 
  },
  If[ Head[labels] != List,
    Return[False]
  ];
  cp = Count[labels, 1];
  cm = Count[labels, -1];
  If[cp + cm != Length[labels],
    Message[svm::wrongLabelSymbols]
  ];
  If[ cm == 0 || cp == 0,
    Message[svm::unbalancedLabels]
  ];
  Return[ cp + cm == Length[labels] && cm > 0 && cp > 0 ];
];

svmUnbalancedClassificationLabelsQ[
    labels_  (* labels list to be chekced *)
  ]:=Block[
  {cp, (* number of 1s *)
   cm  (* number of -1s *) 
  },
  If[ Head[labels] != List,
    Return[False]
  ];
  cp = Count[labels, 1];
  cm = Count[labels, -1];
  If[cp + cm != Length[labels],
    Message[svm::wrongLabelSymbols]
  ];
  
  Return[ cp + cm == Length[labels] ];
];

(* svmRegressionLabelsQ checks wether or not the specified argument
   is a list of regression labels, that is each element in the list
   itself is a number.

   returns: True if the specified argument is a list of regression labels,
            False otherwise.
*)
svmRegressionLabelsQ[labels_]:=Head[labels]==List && Max[Length/@labels]==0

(* svmPatternsQ checks wether or not the specified argument
   is a list of patterns, that is each element in the list itself
   has the same length.

   returns: True if the specified argument is a list of patterns,
            False otherwise.
*)
svmPatternsQ[patterns_]:=Block[
  {},
  If[ Head[patterns] != List,
    Return[False]
  ];
  Return[ (Length/@patterns//Variance) == 0 ];
];

(* svmGetKernel takes the definition of a kernel and returns a list of the
   corresponding definitions in Mathematica, AMPL and Python. The argument
   can be:
  "linear"                                            linear kernel
  {"polynomial", p}                                   polynomial kernel of degree p
  {"polynomialHomogeneous", p}                        homogeneous polynomial kernel of degree p
  {"gaussian", sigma}                                 gaussian kernel of standard deviation sigma
  {"hyperbolic", k, c}                                NN kernel having multiplicative and additive paramenters set to k and c
  {"custom", mathVersion, amplVersion, pythonVersion} custom kernel

   returns: a list containing, respectively, the Mathematica, AMPL and Python version of the kernel funcion.
*)
svmGetKernel[
    "linear"  (* custom definition for linear kernel (as it does not require parameters) *)
  ]=
  {Dot,
   "sum{k in 1..n}x[i,k]*x[j,k];",
   "from numpy import dot\ndef kernel(x1, x2):\n\treturn dot(x1,x2)\n",
   " "};

svmGetKernel[
    kernelDesc_List  (* kernel description *)
  ]:=Block[
  {p,     (* degree in polynomial *)
   sigma, (* standard deviation in gaussian *)
   k,     (* multiplier in hyperbolic *)
   q      (* additive term in hyperbolic *)
  },
	Which[
		kernelDesc[[1]] == "polynomial" && Length[kernelDesc] == 2,
			p = kernelDesc[[2]];
			Return[{Evaluate[(#1.#2+1)^p]&,
 	       "(sum{k in 1..n}x[i,k]*x[j,k]+1)^"<>ToString[p]<>";",
			"from numpy import dot\ndef kernel(x1, x2):\n\treturn (dot(x1,x2)+1)**"<>ToString[p]<>"\n",
			" -t 1 -d "<>ToString[p]<>" -r 1 -s 1"}];,
		kernelDesc[[1]] == "polynomialHomogeneous" && Length[kernelDesc] == 2,
			p = kernelDesc[[2]];
		    Return[{Evaluate[(#1.#2)^p]&,
			"(sum{k in 1..n}x[i,k]*x[j,k])^"<>ToString[p]<>";",
			"from numpy import dot\ndef kernel(x1, x2):\n\treturn (dot(x1,x2))**"<>ToString[p]<>"\n",
			"-t 1 -d "<>ToString[p]<>" -r 0 -s 1 "}];,
		 kernelDesc[[1]] == "gaussian" && Length[kernelDesc]==2,
			sigma = kernelDesc[[2]];
			
			Return[{Evaluate[Exp[-1*Norm[#2-#1]^2/(2sigma^2)]]&,
			"exp(-1*(sum{k in 1..n}(x[i,k]-x[j,k])^2)/(2*"<>ToString[sigma^2]<>"));\n",
			"from numpy import array,dot, exp\n"<>
				"def kernel(x1, x2):\n"<>
				"\tx=array([x1[i]-x2[i] for i in range(len(x1))])\n"<>
				"\treturn exp(-1*dot(x,x.conj())/(2*"<>ToString[sigma^2]<>"))\n",
			"-t 2 -g "<>ToString[ (2^.5 sigma)^-2]}];,
		kernelDesc[[1]] == "hyperbolic" && Length[kernelDesc]==3,
			k = kernelDesc[[2]];
			q = kernelDesc[[3]];
			Return[{Evaluate[Tanh[k #1.#2 +q]]&,
			"tanh("<>ToString[k]<>" * (sum{k in 1..n}x[i,k]*x[j,k]) + "<>ToString[q]<>");",
			"from numpy import dot, tanh\n"<>
				"def kernel(x1, x2):\n"<>
				"\treturn tanh("<>ToString[k]<>" * dot(x1, x2) + "<>ToString[q]<>")",
			"-t 3 -s "<>ToString[k]<>" -r "<>ToString[q]}];,
		kernelDesc[[1]] == "custom" && Length[kernelDesc] == 4 && Head[kernelDesc[[2]]] == Function 
		&& Head[kernelDesc[[3]]] == String && Head[kernelDesc[[4]]] == String,
			Return[{kernelDesc[[2]],kernelDesc[[3]],kernelDesc[[4]]}];
	   ];
	Message[svm::unknownKernel,kernelDesc];
];
(*svmGetKernel[
    kernelDesc_List  (* kernel description *)
  ]:=Block[
  {p,     (* degree in polynomial *)
   sigma, (* standard deviation in gaussian *)
   k,     (* multiplier in hyperbolic *)
   q      (* additive term in hyperbolic *)
  },
  If[ kernelDesc[[1]] == "polynomial" && Length[kernelDesc] == 2,
    p = kernelDesc[[2]];
    Return[{
      Evaluate[(#1.#2+1)^p]&,
      "(sum{k in 1..n}x[i,k]*x[j,k]+1)^"<>ToString[p]<>";",
      "from numpy import dot\ndef kernel(x1, x2):\n\treturn (dot(x1,x2)+1)**"<>ToString[p]<>"\n",
	  " -t 1 -d "<>ToString[p]<>"-r 1 -s 1"}
    ];
  ];
  If[ kernelDesc[[1]] == "polynomialHomogeneous" && Length[kernelDesc] == 2,
    p = kernelDesc[[2]];
    Return[{
      Evaluate[(#1.#2)^p]&,
      "(sum{k in 1..n}x[i,k]*x[j,k])^"<>ToString[p]<>";",
      "from numpy import dot\ndef kernel(x1, x2):\n\treturn (dot(x1,x2))**"<>ToString[p]<>"\n",
      "-t 1 -d "<>ToString[p]<>" -r 0 -s 1 "}
    ];
  ];
  If[ kernelDesc[[1]] == "gaussian" && Length[kernelDesc]==2,
    sigma = kernelDesc[[2]];
    Return[{
      Evaluate[Exp[-1*Norm[#2-#1]^2/(2sigma^2)]]&,
      "exp(-1*(sum{k in 1..n}(x[i,k]-x[j,k])^2)/(2*"<>ToString[sigma^2]<>"));\n",
      "from numpy import array,dot, exp\n"<>
        "def kernel(x1, x2):\n"<>
        "\tx=array([x1[i]-x2[i] for i in range(len(x1))])\n"<>
        "\treturn exp(-1*dot(x,x.conj())/(2*"<>ToString[sigma^2]<>"))\n",
	  "-t 2 -g "<>ToString[sigma]}
	];
  ];
  If[ kernelDesc[[1]] == "hyperbolic" && Length[kernelDesc]==3,
    k = kernelDesc[[2]];
    q = kernelDesc[[3]];
    Return[{
      Evaluate[Tanh[k #1.#2 +q]]&,
      "tanh("<>ToString[k]<>" * (sum{k in 1..n}x[i,k]*x[j,k]) + "<>ToString[q]<>");",
      "from numpy import dot, tanh\n"<>
        "def kernel(x1, x2):\n"<>
        "\treturn tanh("<>ToString[k]<>" * dot(x1, x2) + "<>ToString[q]<>")",
	  "-t 3 -s "<>ToString[k]<>" -r "<>ToString[q]	
	}];
  ];
  If[ kernelDesc[[1]] == "custom" && Length[kernelDesc] == 4 && Head[kernelDesc[[2]]] == Function 
      && Head[kernelDesc[[3]]] == String && Head[kernelDesc[[4]]] == String,
    Return[{
      kernelDesc[[2]],
      kernelDesc[[3]],
      kernelDesc[[4]]
    }];
  ];
  Message[svm::unknownKernel,kernelDesc];
];*)

(* svmFilterOptions is used in order to filter out from a list of replacement
   rules those actually used as options in a particular function.

   Returns: the replacement rules not used as options on a function (to be
   typically used as options for another function).
*)
svmFilterOptions[
  specifiedOptions_,  (* list of replacement rules to be filtered *)
  standardOptions_    (* list of replacement rules to be excluded *)
] := Block[
  {allOpts,  (* predecessor of specified replacement rules *)
   stdOpts   (* predecessor of replacement rules to be filtered out *)
  },
  allOpts = #[[1]]& /@ specifiedOptions;
  stdOpts = #[[1]]& /@ standardOptions;
  Return[Select[specifiedOptions,MemberQ[Complement[allOpts,stdOpts],#[[1]]]&]]
];

svmPythonOptimizer[
    q_,  (*  *)
    p_,  (*  *)
    a_,  (*  *)
    b_,
    g_,
    h_,
    opts___
]:=Block[
  {stdin,       (* string containing the on-the-fly generated Python program to be run *)
   input,       (* file containing the Python program to be run *)
   output,      (* file containing the output of Python *)
   retCode,     (* return code of Python *)
   retValue,    (* return value of this function *)
   isVerbose      (* flag triggerning verbose output *)
    (* global svmPythonAvailable: flag triggering Python availability *)
  },
  If[svmPythonAvailable == False,
    Message[svm::PythonUnavailable];
  ];

  stdin = "from cvxopt import matrix\n";
  stdin = stdin <> "from cvxopt import solvers\n";
  stdin = stdin <> "from classificationDefs import kronecker_delta,chop,chop_c,svm_classification,svm_classification_c\n\n";
  stdin = stdin <> "solvers.options['show_progress']=False\n";
  stdin = stdin <> "solvers.options['maxiters']=1000\n";
  stdin = stdin <> "solvers.options['solver']='mosek'\n\n";

  stdin = stdin <> "Q=matrix(" <> StringReplace[ToString[AccountingForm[q//N, NumberSigns->{"-", ""}]], {"{"->"[", "}"->"]"}] <> ")\n";
  stdin = stdin <> "p=matrix(" <> StringReplace[ToString[AccountingForm[p//N, NumberSigns->{"-", ""}]],{"{"->"[", "}"->"]"}] <> ")\n";
  stdin = stdin <> "G=matrix(" <> StringReplace[ToString[AccountingForm[g//N//Transpose, NumberSigns->{"-", ""}]],{"{"->"[", "}"->"]"}] <> ")\n";
  stdin = stdin <> "h=matrix(" <> StringReplace[ToString[AccountingForm[h//N, NumberSigns->{"-", ""}]],{"{"->"[", "}"->"]"}] <> ")\n";
  stdin = stdin <> "A=matrix(" <> StringReplace[ToString[AccountingForm[a//N, NumberSigns->{"-", ""}]],{"{"->"[", "}"->"]"}] <> ", (1, " <> ToString[Length[a]] <> "))\n";
  stdin = stdin <> "b=matrix(0.0)\n";
  stdin = stdin <> "sol=solvers.qp(Q,p,G,h,A,b)\n";

  stdin = stdin <> "print \"{\",\n";
  stdin = stdin <> "for s in sol['x'][:-1]:\n";
  stdin = stdin <> "\tprint \"%.10f\" % s, \", \",\n";
  stdin = stdin <> "print \"%.10f\" % sol['x'][-1],\"}\"";

  isVerbose = verbose /. {opts} /. Options[svmRegression];
  If[ isVerbose,
    Print[stdin]
  ];

  input = OpenWrite[];
  WriteString[input,stdin];
  Close[input];
  output = OpenWrite[];
  Close[output];
  Run["/bin/ln -sf " <> $UserBaseDirectory <> "/Applications/svMathematica/classificationDefs.py /tmp/classificationDefs.py" ];
  retCode = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> ":$PATH;python " <> input[[1]] <> " > "<>output[[1]]];
  retValue=If[ retCode == 0,
    ReadList[output[[1]], Record][[-1]]//ToExpression,
    $Failed
  ];
  DeleteFile[input[[1]]];
  DeleteFile[output[[1]]];
  If[isVerbose,
    Print[input[[1]]];
    Print[output[[1]]];
  ];
  Return[retValue];
];

svmChop[x_,c_]:=Block[{tol},
  tol = 10^-4;
  If[x<tol,Return[0]];
  If[c-x<tol,Return[c]];
  Return[x];
]











(* ::Section::Closed:: *)
(*Graphics*)


(* svmSamplePlot draws the scatter plot of a sample used for SVM classification.
   The function works only for 1, 2 and 3D patterns and automatically finds out
   which kind of graphic object has to be generated.

   Returns: the Graphics or Graphics3D object corresponding to a scatter plot of the sample
*)
svmClassificationSamplePlot[
    patterns_?svmPatternsQ,            (* labels affecting patterns appearance *)
    labels_?svmUnbalancedClassificationLabelsQ,  (* patterns to be drawn *)
    opts___
  ]:=Block[
  {colorPos,    (* colour of positive points *)
   colorNeg,    (* colour of negative points *)
   sizePos,     (* size of positive points *)
   sizeNeg,     (* size of negative points *)
   allOpts,     (* specified options *)
   stdOpts,     (* function-specific options *)
   n,           (* pattern dimension *)
   grFunc,      (* function used to render graphics *)
   locPatterns, (* editable copy of patterns *)
   furtherOpts, (* additional options to be forwarded to Graphics (3D) *)
   colors       (* list containing the colors and graphics specifications for each point in the scatter plot *)
  },
  n = patterns[[1]]//Length;
  If[n>3,
    Message[svm::invalidPatternDimension];
  ];

  colorPos = positiveColor /. {opts} /. Options[svmClassificationSamplePlot];
  colorNeg = negativeColor /. {opts} /. Options[svmClassificationSamplePlot];
  sizePos = positiveSize /. {opts} /. Options[svmClassificationSamplePlot];
  sizeNeg = negativeSize /. {opts} /. Options[svmClassificationSamplePlot];
  allOpts = #[[1]]& /@ {opts};
  stdOpts = #[[1]]& /@ Options[svmClassificationSamplePlot];
  If[n==1,
    locPatterns = Append[#,0]& /@ patterns;
    n=2,
  (* else *)
    locPatterns=patterns;
 ];
  grFunc = If[n==2, Graphics, Graphics3D];
  furtherOpts = svmFilterOptions[{opts}, Options[svmClassificationSamplePlot]];
  colors = If[#==1,
            {colorPos, AbsolutePointSize[sizePos]},
            {colorNeg, AbsolutePointSize[sizeNeg]}
           ]&/@labels;


  Return[
    grFunc[
      Append[#[[1]], #[[2]]]&/@
      Transpose[
        {
          colors,
          Point /@ locPatterns
        }
      ], Sequence[furtherOpts]
    ]
  ]
]





(* svmSamplePlot draws the scatter plot of a sample used for SVM regression.
   The function works only for 1, 2 and 3D patterns and automatically finds out
   which kind of graphic object has to be generated.

   Returns: the Graphics or Graphics3D object corresponding to a scatter plot of the sample
*)
svmRegressionSamplePlot[
    patterns_?svmPatternsQ,            (* labels affecting patterns appearance *)
    labels_?svmRegressionLabelsQ,  (* patterns to be drawn *)
    opts___
  ]:=Block[
  {n,           (* pattern dimension *)
   locPatterns
  },
  n = patterns[[1]]//Length;
  If[n>2,
    Message[svm::invalidPatternDimension];
  ];

  locPatterns = Append[#[[1]],#[[2]]]&/@Transpose[{patterns, labels}];

  Return[svmClassificationSamplePlot[locPatterns, Table[1,{Length[labels]}], opts]];
]

(* svmCheckColors checks whether or not two color function specifications
   are valid (i.e. either RGBColor, GrayLevel or Hue) and identical. If this
   does not hold an appropriate Message is thrown.

   Returns: the color function specificiation in positive case, $Failed otherwise.
*)
svmCheckColors[
    positiveColor_,  (* color specification for label 1 *)
    negativeColor_   (* color specification for label -1 *)
  ]:=Block[
  {},
  If[Head[positiveColor] != Head[negativeColor],
    Message[svm::unbalancedColors];
    Return[$Failed];
 ];
  If[Head[positiveColor] != RGBColor || Head[positiveColor] != GrayLevel || Head[positiveColor] != Hue,
    Message[svm::invalidColor];
    Return[$Failed];
 ];
 Return[Head[positiveColor]];
]

(* svmDecisionFunctionPlot plots the decision surface corresponding to an SVM
   classifier. The function is overloaded in order to work with 1, 2, and 3D
   patterns.

   Returns: the Graphics or Graphics3D object corresponding to a decision
            function (possibly coupled with margin indication)

*)
svmDecisionFunctionPlot[
    decisionFunction_,    (* Mathematica function whose zeroes describe the frontier between positive and negative subspace *)
    {x_, xLow_, xHigh_},  (* Plot-like specification of independent variable and corresponding lower and upper limits *)
    opts___               (* possible options to be evaluated *)
  ]:= Block[
  {isFrontier,         (* flag deciding whether or not to draw the frontier between subspaces *)
   frontierStyleSp,    (* graphic style to be used for the frontier *)
   grFrontier,       (* graphic object corresponding to the frontier *)
   isShading,          (* flag deciding whether or not shading the values of decision function *)
   grShade,          (* graphics object corresponding to decision function shading *)
   shadingContoursSp,  (* number of contours to be drawn in decision function shading *)
   positiveColor,    (* color for positive subspace *)
   negativeColor,    (* color for negative subspace *)
   colorType,        (* color function (RGBColor, Hue or GrayLevel) to be used in decision function shading *)
   colorFunction,    (* value to be used for the ColorFunction option of ContourPlot *)
   isMargin,           (* flag deciding whether or not to draw margins *)
   marginStyleSp,      (* graphic style to be used for margins *)
   grMargin,         (* graphic object corresponding to margins *)
   furtherOpts       (* further options to be possibly forwarded to Show *)
  },
  furtherOpts = svmFilterOptions[{opts}, Options[svmDecisionFunctionPlot]];
  isFrontier = frontier/.{opts}/.Options[svmDecisionFunctionPlot];
  frontierStyleSp = frontierStyle/.{opts}/.Options[svmDecisionFunctionPlot];
  grFrontier = If[isFrontier,
    ContourPlot[decisionFunction[{x}],{x,xLow,xHigh},{y,-0.5,.5},Contours->{0},
      ContourShading->False,ContourStyle->frontierStyleSp],
    {}
  ];
  isShading = shading /. {opts} /. Options[svmDecisionFunctionPlot];
  {positiveColor, negativeColor} = shadingColors /. {opts} /. Options[svmDecisionFunctionPlot];
  shadingContoursSp = shadingContours /. {opts} /. Options[svmDecisionFunctionPlot];
  colorType = svmCheckColors[positiveColor, negativeColor];
  colorFunction = colorType[List@@positiveColor# + List@@negativeColor(1-#)]&;
  grShade = If[isShading,
    ContourPlot[decisionFunction[{x}], {x,xLow,xHigh}, {y,-.5,.5},
      Contours->shadingContoursSp, ContourLines->False, ColorFunction->colorFunction],
    {}
  ];
  isMargin = margin /. {opts} /. Options[svmDecisionFunctionPlot];
  marginStyleSp = marginStyle /. {opts} /. Options[svmDecisionFunctionPlot];
  grMargin=If[isMargin,
    ContourPlot[decisionFunction[{x}], {x,xLow,xHigh}, {y,-.5,.5},
      Contours->{-1,1}, ContourShading->False, ContourStyle->marginStyleSp],
    {}
  ];
  Return[Show[grShade, grFrontier, grMargin, Sequence@@furtherOpts]];
]

svmDecisionFunctionPlot[
    decisionFunction_,    (* Mathematica function whose zeroes describe the frontier between positive and negative subspace *)
    {x_, xLow_, xHigh_},  (* Plot-like specification of first independent variable and corresponding lower and upper limits *)
    {y_, yLow_, yHigh_},  (* Plot-like specification of second independent variable and corresponding lower and upper limits *)
    opts___               (* possible options to be evaluated *)
  ]:= Block[
  {isFrontier,          (* flag deciding whether or not to draw the frontier between subspaces *)
   frontierStyleSp,     (* graphic style to be used for the frontier *)
   grFrontier,        (* graphic object corresponding to the frontier *)
   isShading,           (* flag deciding whether or not shading the values of decision function *)
   grShade,           (* graphics object corresponding to decision function shading *)
   shadingContoursSp,   (* number of contours to be drawn in decision function shading *)
   positiveColor,     (* color for positive subspace *)
   negativeColor,     (* color for negative subspace *)
   colorType,         (* color function (RGBColor, Hue or GrayLevel) to be used in decision function shading *)
   colorFunction,     (* value to be used for the ColorFunction option of ContourPlot *)
   isMargin,            (* flag deciding whether or not to draw margins *)
   marginStyleSp,       (* graphic style to be used for margins *)
   grMargin,          (* graphic object corresponding to margins *)
   furtherOpts        (* further options to be possibly forwarded to Show *)
  },
  furtherOpts = svmFilterOptions[{opts}, Options[svmDecisionFunctionPlot]];
  isFrontier = frontier /. {opts} /. Options[svmDecisionFunctionPlot];
  frontierStyleSp = frontierStyle /. {opts} /. Options[svmDecisionFunctionPlot];
  grFrontier = If[isFrontier,
    ContourPlot[decisionFunction[{x,y}], {x,xLow,xHigh}, {y,yLow,yHigh},
      Contours->{0}, ContourShading->False, ContourStyle->frontierStyleSp],
    {}
  ];
  isShading = shading /. {opts} /. Options[svmDecisionFunctionPlot];
  {positiveColor, negativeColor} = shadingColors /. {opts} /. Options[svmDecisionFunctionPlot];
  shadingContoursSp = shadingContours /. {opts} /. Options[svmDecisionFunctionPlot];
  colorType = svmCheckColors[positiveColor, negativeColor];
  colorFunction = colorType[List@@positiveColor# + List@@negativeColor(1-#)]&;
  grShade = If[isShading,
    ContourPlot[decisionFunction[{x,y}], {x,xLow,xHigh}, {y,yLow,yHigh},
      Contours->shadingContoursSp, ContourLines->False, ColorFunction->colorFunction],
    {}
  ];
  isMargin = margin /. {opts} /. Options[svmDecisionFunctionPlot];
  marginStyleSp = marginStyle /. {opts} /. Options[svmDecisionFunctionPlot];
  grMargin=If[isMargin,
    ContourPlot[decisionFunction[{x,y}], {x,xLow,xHigh}, {y,yLow,yHigh},
      Contours->{-1,1}, ContourShading->False, ContourStyle->marginStyleSp],
    {}
  ];
  Return[Show[grShade, grFrontier, grMargin, Sequence@@furtherOpts]];
]

svmDecisionFunctionPlot[
    decisionFunction_,     (* Mathematica function whose zeroes describe the frontier between positive and negative subspace *)
    {x_, xLow_, xHigh_},   (* Plot-like specification of first independent variable and corresponding lower and upper limits *)
    {y_, yLow_, yHigh_},   (* Plot-like specification of second independent variable and corresponding lower and upper limits *)
    {z_, zLow_, zHigh_},   (* Plot-like specification of third independent variable and corresponding lower and upper limits *)
    opts___                (* possible options to be evaluated *)
  ]:= Block[
  {isFrontier,       (* flag deciding whether or not to draw the frontier between subspaces *)
   frontierStyleSp,  (* graphic style to be used for the frontier *)
   grFrontier,     (* graphic object corresponding to the frontier *)
   isMargin,         (* flag deciding whether or not to draw margins *)
   marginStyleSp,    (* graphic style to be used for margins *)
   grMargin,       (* graphic object corresponding to margins *)
   furtherOpts     (* further options to be possibly forwarded to Show *)
  },
  furtherOpts = svmFilterOptions[{opts}, Options[svmDecisionFunctionPlot]];
  isFrontier = frontier /. {opts} /. Options[svmDecisionFunctionPlot];
  frontierStyleSp = frontier3DStyle /. {opts} /. Options[svmDecisionFunctionPlot];
  grFrontier = If[isFrontier,
    ContourPlot3D[decisionFunction[{x,y,z}], {x,xLow,xHigh}, {y,yLow,yHigh}, {z,zLow,zHigh},
      Contours->{0}, ContourStyle->frontierStyleSp],
    {}
  ];

  isMargin = margin /. {opts} /. Options[svmDecisionFunctionPlot];
  marginStyleSp = margin3DStyle /. {opts} /. Options[svmDecisionFunctionPlot];
  grMargin=If[isMargin,
    ContourPlot3D[decisionFunction[{x,y,z}], {x,xLow,xHigh}, {y,yLow,yHigh}, {z,zLow,zHigh},
      Contours->{-1,1}, ContourStyle->marginStyleSp],
    {}
  ];
  Return[Show[grFrontier, grMargin, Sequence@@furtherOpts]];
]











(* ::Section:: *)
(*Classification*)


(* svmClassification is the interface method called in order to
   learn a SVM classifier. The method forwards its arguments to
   a chosen implementation. Each implementation basically specifies
   a different way to solve the constrained optimization problem
   at the core of SVM classification.

   Returns: a list of the optimal values for the SVM classification optimization problem.
*)
svmClassification[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmClassificationLabelsQ,  (* example labels *)
    opts___                            (* options to be possibly parsed *)
  ]:=Block[
  {svmClassFunc  (* implementation to be called *)
  },
    If[Length[patterns] != Length[labels],
      Message[svm::unequalLength, patterns, labels];
    ];
    svmClassFunc = implementation /. {opts} /. Options[svmClassification];
    Return[svmClassFunc[patterns, labels, opts]];
  ];

(* svmClassificationMaximize is an implementation of svmClassification
   relying on NMaximize in order to solve the optimization problem at
   the core of SVM classification.

   Returns: a list of the optimal values for the SVM classification optimization problem.
*)
svmClassificationMaximize[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmClassificationLabelsQ,  (* example labels *)
    opts___                            (* options to be possibly parsed *)
  ]:=Block[
  {m,
   alpha,       (* independent variables of the optimization problem *)
   kernelDesc,  (* kernel description *)
   kernelF,      (* kernel of the SVM classification algorithm, as a pure function *)
   lagrangian,  (* objective function of the optimization problem *)
   cSp,           (* parameter C of the SVM classification algorithm *)
   constraints, (* list containing the constraints' symbolic description *)
   solution     (* solution returned by NMaximize *)
  },
  
  m = Length[labels];
  alpha = Subscript[\[Alpha], #]& /@ Range[m];
  kernelDesc = kernel /. {opts} /. Options[svmClassification];
  kernelF = svmGetKernel[kernelDesc][[1]];
  lagrangian = Plus@@alpha - 1/2 * (Times@@#& /@ Tuples[alpha labels,2]).(kernelF@@#& /@ Tuples[ patterns ,2]);
  cSp = c /. {opts} /. Options[svmClassification];
  constraints = Join[{alpha.labels == 0}, 0 <= # <= cSp & /@ alpha];
  solution = NMaximize[{lagrangian, constraints}, alpha];
  (* Return[ alpha /. solution[[2]] ]; *)
  Return[ svmGetClassifier[ patterns, labels, alpha /. solution[[2]], opts ] ];
];

(* svmClassificationAMPL is an implementation of svmClassification
   relying on AMPL and SNOPT in order to solve the optimization problem at
   the core of SVM classification.

   Returns: a list of the optimal values for the SVM classification optimization problem.
*)
svmClassificationAMPL[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmClassificationLabelsQ,  (* example labels *)
    opts___                            (* options to be possibly parsed *)
  ]:=Block[
  {stdin,       (* string containing the on-the-fly generated AMPL program to be run *)
   input,       (* file containing the AMPL program to be run *)
   output,      (* file containing the output of AMPL *)
   i,           (* cycle variable *)
   k,           (* cycle variable *)
   kernelDesc,  (* kernel description *)
   kernelStr,      (* kernel of the SVM classification problem, as a string containing AMPL code *)
   cSp,           (* parameter C of the SVM classification algorithm *)
   retCode,     (* return code of AMPL *)
   retValue,    (* return value of this function *)
   m,           (* number of examples to be learnt *)
   n,           (* dimenstion of each pattern *)
   isVerbose      (* flag triggerning verbose output *)
   (* global svmAMPLAvailable: flag triggering AMPL availability *)
  },
  If[svmAMPLAvailable == False,
    Message[svm::AMPLUnavailable];
  ];
  
  m = Length[patterns];
  n = Length[patterns[[1]]];
  kernelDesc=kernel /. {opts} /. Options[svmClassification];
  kernelStr = svmGetKernel[kernelDesc][[2]];
  cSp = c /. {opts} /. Options[svmClassification];
  stdin = "param m integer > 0 default " <> ToString[m]<>"; # number of sample points\n";
  stdin = stdin <> "param n integer > 0 default " <> ToString[n]<>"; # sample space dimension\n";
  If[ cSp < Infinity,
    stdin = stdin <> "param c > 0 default " <> ToString[cSp] <> "; # trade-off constant\n\n";
  ];
  stdin = stdin <> "param x {1..m,1..n}; # sample points\n";
  stdin = stdin <> "param y {1..m}; # sample labels\n";
  stdin = stdin <> "param dot{i in 1..m,j in 1..m}:="<>kernelStr<>"\n\n";
  stdin = stdin <> "var alpha{1..m}>=0";
  If[ cSp < Infinity,
    stdin = stdin <> " <= " <> ToString[cSp]
  ];
  stdin = stdin <> ";\n";
  stdin = stdin <> "maximize quadratic_form:\n";
  stdin = stdin <> "sum{i in 1..m} alpha[i]\n";
  stdin = stdin <> "-1/2*\n";
  stdin = stdin <> "sum{i in 1..m,j in 1..m}alpha[i]*alpha[j]*y[i]*y[j]*dot[i,j];\n\n";
  stdin = stdin <> "subject to linear_constraint:\n";
  stdin = stdin <> "sum{i in 1..m} alpha[i]*y[i]=0;\n\n";
  stdin = stdin <> "data;\n\n";
  stdin = stdin <> "param\tx:\t";
  For[ k = 1, k <= n, k++,
    stdin = stdin <> ToString[k] <> "\t";
  ];
  stdin = stdin <> ":=\n";
  For[ i = 1, i <= m, i++,
    stdin = stdin <> ToString[i] <> "\t";
    For[ k = 1, k <= n, k++,
      stdin = stdin <> ToString[patterns[[i]][[k]]] <> "\t";
    ];
    stdin = stdin <> If[ i == m, ";\n\n", "\n"];
  ];
  stdin = stdin <> "param y :=\n";
  For[ i = 1, i <= m, i++,
    stdin = stdin <> ToString[i] <> "\t" <> ToString[labels[[i]]];
    stdin = stdin <> If[ i == m, ";\n\n", "\n"];
  ];
  stdin = stdin <> "option solver snopt;\n\n";
  stdin = stdin <> "solve;\n\n";
  stdin = stdin <> "printf: \"{\";\n";
  stdin = stdin <> "printf {k in 1..m-1}:\"%f,\",alpha[k];\n";
  stdin = stdin <> "printf: \"%f}\",alpha[n];\n";
 
  isVerbose = verbose /. {opts} /. Options[svmClassification];
  If[ isVerbose,
    Print[stdin]
  ];

  input = OpenWrite[];
  WriteString[input,stdin];
  Close[input];
  output = OpenWrite[];
  Close[output];
  retCode = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> ":$PATH;ampl < " <> input[[1]] <> " > "<>output[[1]]];
  retValue = If[retCode==0,
    ReadList[output[[1]], Record][[-1]]//ToExpression,
    $Failed
  ];
	
  DeleteFile[input[[1]]];
  DeleteFile[output[[1]]];
  Return[ svmGetClassifier[ patterns, labels, retValue, opts ] ];
];

(* svmClassificationSVLight is an implementation of svmClassification
   relying on SVMLight library in order to solve the optimization problem at
   the core of SVM classification.

   Returns: a list of the optimal values for the SVM classification optimization problem.
*)
svmClassificationSVMLight[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmClassificationLabelsQ,  (* example labels *)
    opts___                            (* options to be possibly parsed *)
]:=Block[
  {kernelDesc,  (* kernel description *)
   kernelF,      (* kernel of the SVM classification problem, as a string containing a Python function *)
   cSp,           (* parameter C of the SVM classification algorithm *)
   m,            (* number of examples to be learnt *)
   n,           (* dimension of examples to be learnt *)
   stdin,
   k,
   i,
   input,
   output,
	command,
	retCode,
	retValue,
	isVerbose,
	stringa,
   q,
   p,
   a,
   b,
   g,
   h,
   alpha
  (* global svmPythonAvailable: flag triggering SVMLight availability *)
  },
   If[svmSVMLightAvailable == False,
	Message[svm::SVMLightUnavailable];
];
	stdin="";
	kernelDesc = kernel /. {opts} /. Options[svmClassification];
    kernelF = svmGetKernel[kernelDesc][[4]];
    cSp = c /. {opts} /. Options[svmClassification];
    
    m=Length[patterns];
    n=Length[patterns[[1]]];
    For[i=1,i<=m,i++,
    stdin=stdin<>ToString[labels[[i]]];
    For[k=1,k<=n,k++,
    stdin=stdin<>" "<>ToString[k]<>":"<>ToString[patterns[[i]][[k]]]<>" "
    ];
    stdin=stdin<>"\n";
    ];
	
	input=OpenWrite[];
	WriteString[input,stdin];
	Close[input];
	output=OpenWrite[];
	Close[output];
	command="svm_learn "<>kernelF<>If[cSp!=Infinity," -c "<>ToString[cSp],"-c 1000000 "]<>" -a "<>output[[1]]<>" "<>input[[1]];

	retCode = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> 
				":$PATH;"<>command];
	retValue = If[retCode==0,
    ReadList[output[[1]], Record]//ToExpression,
    $Failed
  ];
	
	
	DeleteFile[input[[1]]];
	DeleteFile[output[[1]]];
	Return[ svmGetClassifier[ patterns, labels,labels * retValue, opts ] ];
  ];


(* svmClassificationPython is an implementation of svmClassification
   relying on Python and cvxopt in order to solve the optimization problem at
   the core of SVM classification
   Returns: a list of the optimal values for the SVM classification optimization problem.
*)
svmClassificationPython[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmClassificationLabelsQ,  (* example labels *)
    opts___                            (* options to be possibly parsed *)
]:=Block[
  {kernelDesc,  (* kernel description *)
   kernelF,      (* kernel of the SVM classification problem, as a string containing a Python function *)
   cSp,           (* parameter C of the SVM classification algorithm *)
   m,           (* number of examples to be learnt *)
   q,
   p,
   a,
   b,
   g,
   h,
   alpha
  (* global svmPythonAvailable: flag triggering Python availability *)
  },
  If[svmPythonAvailable == False,
    Message[svm::PythonUnavailable];
  ];
  
  kernelDesc = kernel /. {opts} /. Options[svmClassification];
  kernelF = svmGetKernel[kernelDesc][[1]];
  q= Outer[kernelF, patterns, patterns, 1] Transpose[{labels}].{labels};
  m = Length[labels];
  p = Table[-1,{m}];
  a = labels;
  b = 0;
  cSp = c /. {opts} /. Options[svmClassification];
  If[cSp<Infinity,
    g = Join[-IdentityMatrix[m],IdentityMatrix[m]];
    h = Join[Table[0, {m}], Table[cSp, {m}]],
  (* else *)
    g = -IdentityMatrix[m];
    h = Table[0, {m}];
 ];
  
  alpha = svmPythonOptimizer[q, p, a, b, g, h, opts];
	
  Return[ svmGetClassifier[ patterns, labels, alpha, opts ] ];

];

(* svmGetCassifier returns a SVM classifier decision function

   Returns: a Mathematica function corresponding to the decision function of a SVM classifier, or the list of
            optimal values for the lagrange multipliers in the realted dual optimization problem
*)
svmGetClassifier[
    patterns_?svmPatternsQ,           (* learnt patterns *)
    labels_?svmClassificationLabelsQ, (* lerant labels *)
    alpha_List,                       (* optimal values of the optimization problem  *)
    opts___                           (* options to be possibly processed *)
  ]:=Block[
  {kernelDesc,     (* kernel description *)
   kernelF,         (* kernel as a Mathematica function *)
   cSp,              (* parameter C of the SVM classification algorithm *)
   pos,            (* indices of support vectors in the optimization problem's solution *)
   bTbl,           (* list containing the optimal value of b computed according to various equations *)
   bVarThresholdSp,  (* threshold on the variance of the above b table *)
   b,              (* optimal value for intercept in the SVM classification algorithm *)
   alphaChopped,
   classifierInputSp,
   classifierOutputSp,
   classifier
  },
  If[ Not[ Length[patterns] == Length[labels] == Length[alpha] ],
    Message[svm::unequalLengthAlpha,patterns,labels,alpha];
  ];
  kernelDesc = kernel /. {opts} /. Options[svmClassification];
  kernelF = svmGetKernel[kernelDesc][[1]];
  cSp = c /. {opts} /. Options[svmClassification];
  alphaChopped = svmChop[#,cSp]& /@ alpha;
  pos = Flatten[Position[alphaChopped, \[Alpha]_ /; 0<\[Alpha]<cSp]];
  bTbl = labels[[#]]& /@ pos - 
         (Function[{i}, kernelF[#, patterns[[i]]]& /@ patterns] /@ pos).(alphaChopped labels);
  bVarThresholdSp = bVarThreshold /. {opts} /. Options[svmClassification];
  If[ Variance[bTbl] > bVarThresholdSp,
    Message[svm::exceedBVarThreshold,bVarThresholdSp,bTbl]
  ];
  b = Mean[bTbl];


  classifierInputSp = classifierInput /. {opts} /. Options[svmClassification];
  classifierOutputSp = classifierOutput /. {opts} /. Options[svmClassification];

  classifier = Which[
    classifierOutputSp=="real",
      Function[{q},
        Evaluate[
          (alphaChopped labels).(Evaluate /@ kernelF[Hold[#], Hold[q]]&/@patterns)+b
        ]
      ]//ReleaseHold,
    classifierOutputSp=="binary",
      Function[{q},
        Evaluate[
          If[(alphaChopped labels).(Evaluate /@ kernelF[Hold[#], Hold[q]]&/@patterns)+b >= 0, 1, -1]
        ]
      ]//ReleaseHold,
    True,
      Message[svm::invalidOptionValue, "classifierOutput", classifierOutputSp]
  ];

  Switch[classifierInputSp,
    "pattern",
      Return[classifier],
    "sequence",
      Return[Evaluate[classifier[{##}]]&],
    "lagrangeMultipliers",
      Return[alpha],
    _,
      Message[svm::invalidOptionValue, "classifierInput", classifierInputSp]
  ]
(*
  Return[
    Function[{q},
      Evaluate[
        (alphaChopped labels).(Evaluate /@ kernelF[Hold[#], Hold[q]]&/@patterns)+b
      ]
    ]//ReleaseHold
  ];
*)
];
















(* ::Section:: *)
(*Regression*)


(* svmRegression is the interface method called in order to
   learn a SVM regressor. The method forwards its arguments to
   a chosen implementation. Each implementation basically specifies
   a different way to solve the constrained optimization problem
   at the core of SVM regression.

   Returns: a list of the optimal values for the SVM regression optimization problem.
*)
svmRegression[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmRegressionLabelsQ,      (* example labels *)
    opts___                            (* options to be possibly parsed *)
  ]:=Block[
  {svmRegrFunc  (* implementation to be called *)
  },
  If[Length[patterns] != Length[labels],
    Message[svm::unequalLength, patterns, labels];
  ];
  svmRegrFunc = implementation /. {opts} /. Options[svmRegression];
  Return[svmRegrFunc[patterns, labels, opts]];
];

svmPythonInverter[
    m_?MatrixQ,
    opts___
  ]:=Block[
  {stdin,
   input,
   output,
   isVerbose,
   retCode,
   retValue
  },

  stdin = "from numpy import array, linalg\n";
  stdin = stdin <> "def print_array(a):\n";
  stdin = stdin <> "\tprint \"{\",\n";
  stdin = stdin <> "\tfor e in a[:-1]:\n";
  stdin = stdin <> "\t\tprint e , \", \",\n";
  stdin = stdin <> "\tprint a[-1], \"}\",\n";

  stdin = stdin <> "a = array(" <> StringReplace[ToString[m//AccountingForm],{"{"->"[", "}"->"]"}] <> ")\n";
  stdin = stdin <> "b = linalg.inv(a)\n";
  stdin = stdin <> "print \"{\",\n";
  stdin = stdin <> "for r in b[:-1]:\n";
  stdin = stdin <> "\tprint_array(r)\n";
  stdin = stdin <> "\tprint \", \",\n";
  stdin = stdin <> "print_array(b[-1])\n";
  stdin = stdin <> "print \"}\"\n";

  isVerbose = verbose /. {opts} /. Options[svmRegression];
  If[ isVerbose,
    Print[stdin]
  ];

  input = OpenWrite[];
  WriteString[input,stdin];
  Close[input];
  output = OpenWrite[];
  Close[output];
  retCode = Run["export PATH=" <> $UserBaseDirectory <> "/Applications/svMathematica/" <> ":$PATH;python " <> input[[1]] <> " > "<>output[[1]]];
  retValue=If[ retCode == 0,
    ReadList[output[[1]], Record][[-1]]//ToExpression,
    $Failed
  ];
  DeleteFile[input[[1]]];
  DeleteFile[output[[1]]];
  If[isVerbose,
    Print[input[[1]]];
    Print[output[[1]]];
  ];
  Return[retValue];

]

svmRidgeRegression[
    patterns_?svmPatternsQ,
    labels_?svmRegressionLabelsQ,
    opts___
]:=Block[
  {lambdaSp,
   m,
   u,
   kernelDesc,
   kernelF,
   ki,
   invFunc,
   kiinv,
   b,
   alpha
  },
  
  lambdaSp = lambda /. {opts} /. Options[svmRidgeRegression];
  m = Length[patterns];
  u = Table[1, {m}];
  kernelDesc = kernel /. {opts} /. Options[svmRegression];
  kernelF = svmGetKernel[kernelDesc][[1]];
  ki = Outer[kernelF, patterns, patterns, 1] + lambdaSp IdentityMatrix[m];
  invFunc = matrixInverter /. {opts} /. Options[svmRidgeRegression];
  kiinv = invFunc[ki];

  b = (u . kiinv . labels) / (u . kiinv . u);

  alpha = 2 lambdaSp kiinv . (labels - b u);
  Return[
    Function[{q},
      Evaluate[
        1/(2 lambdaSp) alpha . (Evaluate /@ kernelF[Hold[#],Hold[q]]&/@patterns)+b
      ]
    ]//ReleaseHold
  ];
];

svmLinearInsensitiveRegression[
    patterns_?svmPatternsQ,            (* example patterns *)
    labels_?svmRegressionLabelsQ,      (* example labels *)
    opts___                            (* options to be possibly parsed *)
]:=Block[
  {kernelDesc,
   kernelF,
   qBase,
   q,
   epsilonSp,
   p,
   m,
   a,
   b,
   g,
   cSp,
   h,
   optimizerF,
   alpha,
   alphaHat,
   pos,
   bTbl,bTblHat,
   bVarThresholdSp
  },
  kernelDesc = kernel /. {opts} /. Options[svmRegression];
  kernelF = svmGetKernel[kernelDesc][[1]];
  qBase = Outer[kernelF, patterns, patterns, 1];
  q = Join[Join[#, -#]& /@ qBase, Join[-#, #]& /@ qBase];
  epsilonSp = epsilon /. {opts} /. Options[svmLinearInsensitiveRegression];
  p = Join[epsilonSp -labels, epsilonSp + labels];
  m = Length[labels];
  a = Join[Table[1, {m}], Table[-1, {m}]];
  b = 0;
  cSp = c /. {opts} /. Options[svmLinearInsensitiveRegression];
  If[cSp<Infinity,
    g = Join[-IdentityMatrix[2m],IdentityMatrix[2m]];
    h = Join[Table[0, {2m}], Table[cSp, {2m}]],
  (* else *)
    g = -IdentityMatrix[2m];
    h = Table[0, {2m}];
 ];
  
  optimizerF = optimizer /. {opts} /. Options[svmLinearInsensitiveRegression];
  {alpha, alphaHat} = Partition[optimizerF[q, p, a, b, g, h, opts], m];


  pos = Flatten[Position[alpha, \[Alpha]_ /; 0<\[Alpha]<cSp]];
  bTbl = labels[[#]]& /@ pos - 
         (Function[{i}, kernelF[#, patterns[[i]]]& /@ patterns] /@ pos).(alpha - alphaHat) - epsilonSp;

  pos = Flatten[Position[alphaHat, \[Alpha]_ /; 0<\[Alpha]<cSp]];
  bTblHat = labels[[#]]& /@ pos - 
         (Function[{i}, kernelF[#, patterns[[i]]]& /@ patterns] /@ pos).(alpha - alphaHat) + epsilonSp;



  bVarThresholdSp = bVarThreshold /. {opts} /. Options[svmRegression];
  If[ Variance[Join[bTbl,bTblHat]] > bVarThresholdSp,
    Message[svm::exceedBVarThreshold,bVarThresholdSp,bTbl]
  ];
  b = Mean[Join[bTbl,bTblHat]];

  Return[
    Function[{q},
      Evaluate[
        (alpha - alphaHat).(Evaluate /@ kernelF[Hold[#], Hold[q]]&/@patterns)+b
      ]
    ]//ReleaseHold
  ];

];











(* ::Section:: *)
(*Package finalization*)


 End[]

(* Protect the public symbols for functions and options *)
Protect[
  c,
  kernel,
  bVarThreshold,
  implementation,
  verbose,
  positiveColor,
  negativeColor,
  positiveSize,
  negativeSize,
  svmClassification,
  svmClassificationMaximize,
  svmClassificationAMPL,
  svmClassificationPython,
  svmClassificationSVMLight,
  svmGetClassifier,
  svmClassificationSamplePlot,
  svmRegressionSamplePlot,
  svmRegression,
  svmRegressionMaximize,
  svmRegressionAMPL,
  svmRegressionPython,
  lambda,
  matrixInverter
];

EndPackage[]






