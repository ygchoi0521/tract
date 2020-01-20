use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct RmDims {
    pub axes: Vec<usize>,
}

impl RmDims {
    fn compute_shape<D: DimLike>(&self, input: &[D]) -> TVec<D> {
        input
            .iter()
            .enumerate()
            .filter(|(ix, _d)| !self.axes.contains(ix))
            .map(|(_ix, d)| d.clone())
            .collect()
    }

    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(&self, input: Arc<Tensor>) -> TractResult<TVec<Arc<Tensor>>> {
        let shape = self.compute_shape(input.shape());
        Ok(tvec![input.into_tensor().into_array::<T>()?.into_shape(&*shape)?.into_arc_tensor()])
    }
}

impl Op for RmDims {
    fn name(&self) -> Cow<str> {
        "RmDims".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {:?}", self.axes)])
    }

    canonic!();
    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for RmDims {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        let input = args_1!(inputs);
        dispatch_datum!(Self::eval_t(input.datum_type())(self, input))
    }
}

impl InferenceRulesOp for RmDims {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, &inputs[0].datum_type)?;
        s.equals(&outputs[0].rank, (&inputs[0].rank).bex() - self.axes.len() as i32)?;
        for axis in &self.axes {
            s.equals(&inputs[0].shape[*axis], 1.to_dim())?;
        }
        s.given(&inputs[0].shape, move |s, shape| {
            let output_shape = self.compute_shape(&shape);
            s.equals(&outputs[0].shape, output_shape)
        })
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for RmDims {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(TypedFact::dt_shape(
            inputs[0].datum_type,
            self.compute_shape(&*inputs[0].shape.to_tvec()).as_ref(),
        )?))
    }

    fn invariants(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Invariants> {
        let mut out = 0;
        let mut axes = tvec!();
        for in_ in 0..model.outlet_fact(node.inputs[0])?.shape.rank() {
            if !self.axes.contains(&out) {
                axes.push(AxisInfo {
                    inputs: tvec!(Some(in_)),
                    outputs: tvec!(Some(out)),
                    period: 1,
                    disposable: true,
                });
                out += 1;
            }
        }
        Ok(axes.into_iter().collect())
    }

    fn dispose_dummy_axis(
        &self,
        _model: &TypedModel,
        _node: &TypedNode,
        axes: &[Option<usize>]
    ) -> TractResult<Option<Box<dyn TypedOp>>> {
        let axis = axes[0].unwrap();
        let axes = self
            .axes
            .iter()
            .cloned()
            .filter(|&a| a != axis)
            .map(|a| a - (a > axis) as usize)
            .collect();
        Ok(Some(Box::new(RmDims::new(axes))))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        'axis: for &rm_axis in &self.axes {
            let tracking = crate::ops::invariants::AxisTracking::for_outlet_and_axis(
                model,
                node.inputs[0],
                rm_axis,
            )?;
            if tracking.creators.iter().any(|c| {
                !model
                    .node(c.node)
                    .op_as::<super::AddDims>()
                    .map(|ad| ad.axes.contains(&tracking.outlets[c]))
                    .unwrap_or(false)
            }) || !tracking.destructors.iter().all(|c| {
                model
                    .node(c.node)
                    .op_as::<RmDims>()
                    .map(|ad| ad.axes.contains(&tracking.outlets[&model.node(c.node).inputs[0]]))
                    .unwrap_or(false)
            }) {
                continue 'axis;
            }
            println!("Decluttering {}", node);
            let mut patch = TypedModelPatch::default();
            let mut mapping = HashMap::<OutletId, OutletId>::new();
            for c in &tracking.creators {
                let node = model.node(c.node);
                let axis = tracking.outlets[&c];
                if let Some(add) = node.op_as::<super::AddDims>() {
                    if add.axes.contains(&axis) {
                        let mut wire = patch.tap_model(model, node.inputs[0])?;
                        if add.axes.len() > 1 {
                            let axes = add
                                .axes
                                .iter()
                                .cloned()
                                .filter(|&a| a != axis)
                                .map(|a| a - (a > axis) as usize)
                                .collect();
                            wire = patch.wire_node(
                                &*node.name,
                                super::AddDims::new(axes),
                                [wire].as_ref(),
                            )?[0];
                        }
                        mapping.insert(node.id.into(), wire);
                    } else {
                        unreachable!();
                    }
                } else {
                    unreachable!();
                }
            }
            for n in model.eval_order()? {
                let node = model.node(n);
                println!("   -> {}", node);
                for i in &node.inputs {
                    if !mapping.contains_key(&i) {
                        mapping.insert(*i, patch.tap_model(model, *i)?);
                    }
                }
                let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
                let axis = if let Some(axis) =
                    node.inputs.get(0).and_then(|input| tracking.outlets.get(input).cloned())
                {
                    axis
                } else {
                    continue;
                };
                let op = node
                    .op
                    .dispose_dummy_axis(model, node, &[Some(axis)])?
                    .unwrap_or_else(|| node.op.clone());
                let outputs = patch.wire_node(&*node.name, op, &*inputs)?;
                for (ix, o) in outputs.into_iter().enumerate() {
                    mapping.insert(OutletId::new(node.id, ix), o);
                }
            }
            for des in tracking.destructors {
                let node = model.node(des.node);
                let axis = *tracking.outlets.get(&node.inputs[0]).unwrap();
                let axes = self
                    .axes
                    .iter()
                    .cloned()
                    .filter(|&a| a != axis)
                    .map(|a| a - (a > axis) as usize)
                    .collect();
                let wire =
                    patch.wire_node(&*node.name, RmDims::new(axes), &[mapping[&node.inputs[0]]])?
                        [0];
                patch.shunt_outside(node.id.into(), wire)?;
            }
        }
        Ok(None)
    }

    fn pulsify(
        &self,
        _source: &NormalizedModel,
        node: &NormalizedNode,
        target: &mut PulsedModel,
        mapping: &HashMap<OutletId, OutletId>,
        _pulse: usize,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        target.wire_node(&*node.name, self.clone(), &[input])
    }

    typed_op_as_op!();
}

impl PulsedOp for RmDims {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape = self.compute_shape(&*inputs[0].shape);
        fact.axis -= self.axes.iter().filter(|&ax| *ax <= fact.axis).count();
        Ok(tvec!(fact))
    }

    pulsed_op_as_op!();
    pulsed_op_to_typed_op!();
}
