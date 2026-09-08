! file: twoelint_core_mod.f90
!
! Shared helpers for TwoElInt zero-copy kernels.
module twoelint_core_mod
  implicit none
  private
  public :: u16_from_i16, get2, gen_perms

contains

  pure integer function u16_from_i16(x) result(v)
    integer*2, intent(in) :: x
    v = int(x)
    if (v < 0) v = v + 65536
  end function u16_from_i16

  pure double precision function get2(a, ncol, r0, c0) result(x)
    double precision, intent(in) :: a(:)
    integer, intent(in) :: ncol
    integer, intent(in) :: r0, c0
    x = a(int(r0, 8) * int(ncol, 8) + int(c0, 8) + 1_8)
  end function get2

  ! Exact port of reorganize_indexes() used by AAAA/BBBB kernels (Python is source-of-truth).
  subroutine gen_perms(ucount, i, j, k, l, nperm, ii, jj, kk, ll)
    implicit none
    integer, intent(in) :: ucount, i, j, k, l
    integer, intent(out) :: nperm
    integer, intent(out) :: ii(8), jj(8), kk(8), ll(8)

    integer :: it, jt, kt, lt
    integer :: oi, oj, ok, ol

    oi = i; oj = j; ok = k; ol = l
    it = oi; jt = oj; kt = ok; lt = ol

    select case (ucount)
    case (4)
      nperm = 8
      ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
      ii(2)=kt; jj(2)=lt; kk(2)=it; ll(2)=jt
      ii(3)=jt; jj(3)=it; kk(3)=lt; ll(3)=kt
      ii(4)=lt; jj(4)=kt; kk(4)=jt; ll(4)=it
      ii(5)=it; jj(5)=jt; kk(5)=lt; ll(5)=kt
      ii(6)=lt; jj(6)=kt; kk(6)=it; ll(6)=jt
      ii(7)=jt; jj(7)=it; kk(7)=kt; ll(7)=lt
      ii(8)=kt; jj(8)=lt; kk(8)=jt; ll(8)=it
      return

    case (3)
      if (it == jt .or. kt == lt) then
        nperm = 4
        ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
        ii(2)=kt; jj(2)=lt; kk(2)=it; ll(2)=jt
        ii(3)=jt; jj(3)=it; kk(3)=lt; ll(3)=kt
        ii(4)=lt; jj(4)=kt; kk(4)=jt; ll(4)=it
        return
      end if

      ! Else: Python returns the full 8-permutation set.
      nperm = 8
      ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
      ii(2)=kt; jj(2)=lt; kk(2)=it; ll(2)=jt
      ii(3)=jt; jj(3)=it; kk(3)=lt; ll(3)=kt
      ii(4)=lt; jj(4)=kt; kk(4)=jt; ll(4)=it
      ii(5)=it; jj(5)=jt; kk(5)=lt; ll(5)=kt
      ii(6)=lt; jj(6)=kt; kk(6)=it; ll(6)=jt
      ii(7)=jt; jj(7)=it; kk(7)=kt; ll(7)=lt
      ii(8)=kt; jj(8)=lt; kk(8)=jt; ll(8)=it
      return

    case (2)
      if (it == jt .and. it /= kt .and. it /= lt) then
        nperm = 2
        ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
        ii(2)=kt; jj(2)=lt; kk(2)=it; ll(2)=jt
        return
      end if

      if (it == kt .and. it /= jt .and. it /= lt) then
        nperm = 4
        ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
        ii(2)=it; jj(2)=jt; kk(2)=lt; ll(2)=kt
        ii(3)=jt; jj(3)=it; kk(3)=kt; ll(3)=lt
        ii(4)=jt; jj(4)=it; kk(4)=lt; ll(4)=kt
        return
      end if

      ! Else-branch: optional reordering for triple-equal patterns (match Python exactly).
      it = oi; jt = oj; kt = ok; lt = ol

      if (oi == oj .and. oj == ok) then
        ! [i,j,k] all equal -> keep
      else if (oi == oj .and. oj == ol) then
        ! [i,j,l] all equal -> (i,j,l,k)
        kt = ol
        lt = ok
      else if (oi == ok .and. ok == ol) then
        ! [i,k,l] all equal -> (i,k,l,j)
        jt = ok
        kt = ol
        lt = oj
      else if (oj == ok .and. ok == ol) then
        ! [j,k,l] all equal -> (j,k,l,i)
        it = oj
        jt = ok
        kt = ol
        lt = oi
      end if

      nperm = 4
      ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
      ii(2)=lt; jj(2)=it; kk(2)=jt; ll(2)=kt
      ii(3)=it; jj(3)=jt; kk(3)=lt; ll(3)=kt
      ii(4)=it; jj(4)=lt; kk(4)=jt; ll(4)=kt
      return

    case default
      nperm = 1
      ii(1)=it; jj(1)=jt; kk(1)=kt; ll(1)=lt
      return
    end select
  end subroutine gen_perms


end module twoelint_core_mod
