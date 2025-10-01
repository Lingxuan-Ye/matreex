pub use rayon::prelude::*;

use super::Matrix;
use super::layout::Order;
use crate::error::Result;
use crate::index::Index;

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn par_apply<F>(&mut self, f: F) -> &mut Self
    where
        T: Send,
        F: Fn(&mut T) + Sync + Send,
    {
        self.data.par_iter_mut().for_each(f);
        self
    }

    pub fn par_map<U, F>(self, f: F) -> Result<Matrix<U, O>>
    where
        T: Send,
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        let layout = self.layout.cast()?;
        let data = self.data.into_par_iter().map(f).collect();
        Ok(Matrix { layout, data })
    }

    pub fn par_map_ref<'a, U, F>(&'a self, f: F) -> Result<Matrix<U, O>>
    where
        T: Sync,
        U: Send,
        F: Fn(&'a T) -> U + Sync + Send,
    {
        let layout = self.layout.cast()?;
        let data = self.data.par_iter().map(f).collect();
        Ok(Matrix { layout, data })
    }
}

impl<T, O> Matrix<T, O>
where
    O: Order,
{
    pub fn par_iter_elements(&self) -> impl ParallelIterator<Item = &T>
    where
        T: Sync,
    {
        self.data.par_iter()
    }

    pub fn par_iter_elements_mut(&mut self) -> impl ParallelIterator<Item = &mut T>
    where
        T: Send,
    {
        self.data.par_iter_mut()
    }

    pub fn into_par_iter_elements(self) -> impl ParallelIterator<Item = T>
    where
        T: Send,
    {
        self.data.into_par_iter()
    }

    pub fn par_iter_elements_with_index(&self) -> impl ParallelIterator<Item = (Index, &T)>
    where
        T: Sync,
    {
        let stride = self.stride();
        self.data
            .par_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }

    pub fn par_iter_elements_mut_with_index(
        &mut self,
    ) -> impl ParallelIterator<Item = (Index, &mut T)>
    where
        T: Send,
    {
        let stride = self.stride();
        self.data
            .par_iter_mut()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }

    pub fn into_par_iter_elements_with_index(self) -> impl ParallelIterator<Item = (Index, T)>
    where
        T: Send,
    {
        let stride = self.stride();
        self.data
            .into_par_iter()
            .enumerate()
            .map(move |(index, element)| {
                let index = Index::from_flattened::<O>(index, stride);
                (index, element)
            })
    }
}
